"""
rec_api.py — Rec v1 Live API (Task 17)

Exposes the PlaceMux Recommendation v1 engine as a live FastAPI service.
Endpoints:
  GET  /health                       — liveness + model status
  POST /recommend/jobs-for-cohort    — top-k jobs per student in a cohort
  POST /recommend/students-for-job   — top-k candidates for a single job
  GET  /dashboard                    — serves the placement dashboard HTML
  GET  /metrics                      — latest Rec v1 summary metrics

Standing instructions: robust error handling, structured logging,
NumPy-style docstrings, random_state=42.
"""

import os
import json
import logging
import time
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from src.model_schemas import StudentFeatures, JobFeatures
from src.ranker import load_ranker
from src.recommender import (
    recommend_jobs_for_cohort,
    recommend_students_for_jobs,
    generate_rec_report,
)
from src.monitor import monitor

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global model state
# ---------------------------------------------------------------------------
_MODEL = None

MODELS_DIR      = "models"
DASHBOARD_PATH  = "logs/dashboard.html"
METRICS_PATH    = "logs/task17_metrics.json"


def _load_best_ranker() -> object:
    """
    Load the best available ranker model (prefers ranker_v2, falls back to v1).

    Returns
    -------
    model : Any
        Loaded LightGBM / sklearn model.

    Raises
    ------
    FileNotFoundError
        If no ranker .pkl is found in models/.
    """
    if not os.path.exists(MODELS_DIR):
        raise FileNotFoundError("models/ directory not found.")
    pkls = [f for f in os.listdir(MODELS_DIR)
            if f.startswith("ranker_") and f.endswith(".pkl")]
    if not pkls:
        raise FileNotFoundError("No ranker model pkl found in models/.")
    pkls.sort(reverse=True)          # ranker_v2 > ranker_v1 lexicographically
    path = os.path.join(MODELS_DIR, pkls[0])
    logger.info(f"[rec_api] Loading model: {path}")
    return load_ranker(path)


# ---------------------------------------------------------------------------
# Lifespan — load model once at startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan: load ranker model on startup, release on shutdown."""
    global _MODEL
    try:
        _MODEL = _load_best_ranker()
        logger.info("[rec_api] Model ready. Rec v1 API is live.")
    except Exception as exc:
        logger.critical(f"[rec_api] STARTUP FAILED — could not load model: {exc}")
        _MODEL = None
    yield
    _MODEL = None
    logger.info("[rec_api] Shutdown complete.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="PlaceMux Rec v1 API",
    description="Placement Recommendation v1 — college cohort job recommendations & candidate shortlists.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------------------------
class CohortRecRequest(BaseModel):
    """
    Request body for cohort job recommendations.

    Parameters
    ----------
    students : List[StudentFeatures]
        College cohort profiles.
    jobs : List[JobFeatures]
        Job postings to rank against.
    top_k : int
        Max jobs to return per student (default 3).
    """
    students: List[StudentFeatures]
    jobs:     List[JobFeatures]
    top_k:    int = Field(default=3, ge=1, le=20)


class JobShortlistRequest(BaseModel):
    """
    Request body for candidate shortlisting per job.

    Parameters
    ----------
    job : JobFeatures
        The target job posting.
    students : List[StudentFeatures]
        Candidate pool.
    top_k : int
        Max candidates to return (default 2).
    """
    job:      JobFeatures
    students: List[StudentFeatures]
    top_k:    int = Field(default=2, ge=1, le=20)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", summary="Liveness + model status check")
def health():
    """
    Return API health and model load status.

    Returns 200 if the model is loaded and ready, 503 otherwise.
    """
    if _MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded — API not ready.")
    return {
        "status":  "ok",
        "service": "PlaceMux Rec v1 API",
        "model":   "loaded",
        "version": "1.0.0",
    }


@app.post("/recommend/jobs-for-cohort", summary="Top-k jobs per student in a cohort")
def api_jobs_for_cohort(req: CohortRecRequest):
    """
    Rank top-k jobs for every student in the request cohort.

    Delegates to recommender.recommend_jobs_for_cohort().
    Fault-isolated per student — one bad profile will not abort the run.

    Parameters
    ----------
    req : CohortRecRequest
        Cohort profiles + job pool + top_k.

    Returns
    -------
    dict
        cohort_recommendations list, college_summary stats, timestamp.
    """
    start_time = time.time()
    if _MODEL is None:
        raise HTTPException(status_code=503, detail="Model is currently unavailable.")
    try:
        if not req.students:
            raise ValueError("students list is empty.")
        if not req.jobs:
            raise ValueError("jobs list is empty.")

        cohort_rec = recommend_jobs_for_cohort(
            model=_MODEL, students=req.students, jobs=req.jobs, top_k=req.top_k
        )
        job_shortlists = recommend_students_for_jobs(
            model=_MODEL, students=req.students, jobs=req.jobs, top_k=2
        )
        report = generate_rec_report(
            students=req.students, jobs=req.jobs,
            cohort_recommendations=cohort_rec, job_shortlists=job_shortlists,
        )
        
        # Track metric for score distribution
        scores = []
        for c in cohort_rec:
            for j in c.get("recommended_jobs", []):
                scores.append(j.get("score", 0.0))
        monitor.record_prediction_scores(scores)
        
        latency_ms = (time.time() - start_time) * 1000
        monitor.record_request(latency_ms, success=True)
        return report

    except ValueError as exc:
        latency_ms = (time.time() - start_time) * 1000
        monitor.record_request(latency_ms, success=False)
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        latency_ms = (time.time() - start_time) * 1000
        monitor.record_request(latency_ms, success=False)
        logger.error(f"/recommend/jobs-for-cohort error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal recommendation error.")


@app.post("/recommend/students-for-job", summary="Top-k candidate shortlist for a job")
def api_students_for_job(req: JobShortlistRequest):
    """
    Shortlist top-k students for a single job posting.

    Parameters
    ----------
    req : JobShortlistRequest
        Target job + candidate pool + top_k.

    Returns
    -------
    dict
        job_id, top_candidates list, avg_candidate_score.
    """
    start_time = time.time()
    if _MODEL is None:
        raise HTTPException(status_code=503, detail="Model is currently unavailable.")
    try:
        if not req.students:
            raise ValueError("students list is empty.")
        results = recommend_students_for_jobs(
            model=_MODEL, students=req.students, jobs=[req.job], top_k=req.top_k
        )
        
        res = results[0] if results else {"job_id": req.job.job_id, "top_candidates": []}
        
        scores = [c.get("score", 0.0) for c in res.get("top_candidates", [])]
        monitor.record_prediction_scores(scores)
        
        latency_ms = (time.time() - start_time) * 1000
        monitor.record_request(latency_ms, success=True)
        return res

    except ValueError as exc:
        latency_ms = (time.time() - start_time) * 1000
        monitor.record_request(latency_ms, success=False)
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        latency_ms = (time.time() - start_time) * 1000
        monitor.record_request(latency_ms, success=False)
        logger.error(f"/recommend/students-for-job error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal shortlist error.")


@app.get("/dashboard", response_class=HTMLResponse, summary="Placement dashboard UI")
def serve_dashboard():
    """
    Serve the generated placement dashboard HTML file.

    Returns
    -------
    HTMLResponse
        The dashboard HTML, or a 404 if not yet generated.
    """
    if not os.path.exists(DASHBOARD_PATH):
        raise HTTPException(
            status_code=404,
            detail="Dashboard not generated yet. Run train_task17.py first."
        )
    with open(DASHBOARD_PATH, "r", encoding="utf-8") as f:
        html = f.read()
    return HTMLResponse(content=html)


@app.get("/metrics", summary="Latest Rec v1 summary metrics")
def get_metrics():
    """
    Return the latest Task 17 summary metrics JSON.

    Returns
    -------
    dict
        Metrics from logs/task17_metrics.json, or 404 if not found.
    """
    if not os.path.exists(METRICS_PATH):
        raise HTTPException(
            status_code=404,
            detail="Metrics not found. Run train_task17.py first."
        )
    with open(METRICS_PATH, "r") as f:
        return json.load(f)

@app.get("/live-metrics", summary="Live Production Model Monitoring Metrics")
def get_live_metrics():
    """
    Returns real-time telemetry from the ModelMonitor.
    Used for live production observability (Task 25).
    """
    return monitor.get_metrics()

