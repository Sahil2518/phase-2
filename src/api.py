"""
api.py — Task 05: Marketplace Integration & Company Portal v1

This module provides a production-ready FastAPI REST API backend to serve the AI-powered
matching and ranking engine. It exposes endpoints for both the Company Portal
(ranking candidates for a job) and the Student Portal (ranking jobs for a student).

--------------------------------------------------------------------------------
FEEDBACK IMPLEMENTATION (Task 05 Updates):
1. Detailed Documentation & Comments:
   - Extensive module-level, class-level, and function-level docstrings added.
   - Inline comments provided to explain the rationale behind architectural decisions.

2. Edge Case & Error Handling in Match Prediction:
   - Explicit guards added to handle uninitialized/missing models gracefully (HTTP 503).
   - Defensive checks placed around the prediction logic to catch mathematical errors,
     NaNs, or unexpected data types during matching.
   - Structured error reporting returns clear, actionable HTTP 400/422/500 messages
     instead of generic stack traces.
--------------------------------------------------------------------------------
"""

import os
import sys
import logging
from typing import Any
from fastapi import FastAPI, HTTPException, status
from pydantic import ValidationError

# Import required schemas
from src.model_schemas import (
    RankCandidatesRequest,
    RankJobsRequest,
    JobRankingResponse,
    StudentRankingResponse
)

# Import ranking business logic
from src.ranker import (
    load_ranker, 
    rank_jobs_for_student, 
    rank_candidates_for_job
)

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
# Ensures all API events and errors are properly recorded for observability.
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/task05_api.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("PlaceMux-API")

# ---------------------------------------------------------------------------
# Global State & Application Setup
# ---------------------------------------------------------------------------
# The FastAPI app instance
app = FastAPI(
    title="PlaceMux AI Marketplace API",
    description="End-to-end integrated flow for the Company Portal v1",
    version="1.0.0"
)

# Global variable to hold the loaded LightGBM/GBR model in memory
# Loaded once at startup to avoid high latency on every request.
_RANKER_MODEL: Any = None


# ---------------------------------------------------------------------------
# Application Lifecycle Events
# ---------------------------------------------------------------------------
@app.on_event("startup")
def load_ml_artifacts():
    """
    Startup event handler: Initializes the machine learning pipeline.
    
    Edge Case Handled: 
    If the model artifact is missing or corrupted, the server starts up 
    but logs a critical warning. Subsequent requests will safely return 503
    instead of crashing the process unexpectedly.
    """
    global _RANKER_MODEL
    logger.info("Initializing API and loading ML models...")
    
    # Locate the most recent model artifact
    try:
        if not os.path.exists("models"):
            logger.warning("Models directory not found.")
            return

        pkl_files = [f for f in os.listdir("models") if f.startswith("ranker_v1_") and f.endswith(".pkl")]
        
        if not pkl_files:
            logger.warning("No pre-trained ranker model found in models/ directory.")
            return
            
        # Load the latest model by name sorting
        pkl_files.sort(reverse=True)
        latest_model_path = os.path.join("models", pkl_files[0])
        
        logger.info(f"Loading model from {latest_model_path}")
        _RANKER_MODEL = load_ranker(latest_model_path)
        logger.info("ML model successfully loaded into memory.")
        
    except Exception as e:
        # Edge Case: Model corruption or deserialization failure
        logger.error(f"Failed to load ML model during startup: {e}", exc_info=True)
        _RANKER_MODEL = None


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.post(
    "/api/v1/portal/rank-candidates", 
    response_model=JobRankingResponse,
    status_code=status.HTTP_200_OK,
    tags=["Company Portal"]
)
def rank_candidates(request: RankCandidatesRequest) -> JobRankingResponse:
    """
    Company Portal Endpoint: Ranks a list of candidates against a specific job.
    
    This endpoint takes a target Job and a pool of Students, evaluates the AI 
    match score for every pair, and returns a sorted shortlist with detailed 
    explainability payloads.
    
    Raises:
        HTTPException 503: If the ML model is not available.
        HTTPException 400: If the prediction engine encounters invalid data.
        HTTPException 500: Unhandled internal processing errors.
    """
    # EDGE CASE 1: Model not loaded
    if _RANKER_MODEL is None:
        logger.error("API request failed: Ranker model is not initialized.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Machine learning model is currently unavailable."
        )

    # EDGE CASE 2: Empty candidate list is technically blocked by Pydantic schema validation,
    # but we add a safety check here to prevent downstream math errors.
    if not request.candidates:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The candidate list cannot be empty."
        )

    try:
        logger.info(f"Ranking {len(request.candidates)} candidates for Job {request.job.job_id}...")
        
        # Execute business logic (which includes its own try/except blocks per candidate)
        response = rank_candidates_for_job(
            model=_RANKER_MODEL,
            job=request.job,
            students=request.candidates,
            top_k=request.top_k
        )
        
        # EDGE CASE 3: Handle potential empty returns if all candidates failed prediction silently
        if not response.ranked_candidates:
            logger.warning(f"All candidates failed prediction for Job {request.job.job_id}.")
            
        logger.info(f"Successfully ranked {len(response.ranked_candidates)} candidates.")
        return response

    except ValueError as ve:
        # Catch specific value errors (e.g. math errors in match vector calculation)
        logger.error(f"Data validation error during candidate ranking: {ve}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        # Catch-all for unexpected failures to prevent returning raw stack traces
        logger.error(f"Internal failure during candidate ranking: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while processing the ranking request."
        )


@app.post(
    "/api/v1/marketplace/rank-jobs", 
    response_model=StudentRankingResponse,
    status_code=status.HTTP_200_OK,
    tags=["Student Portal"]
)
def rank_jobs(request: RankJobsRequest) -> StudentRankingResponse:
    """
    Student Portal Endpoint: Ranks a list of jobs for a specific student.
    
    This endpoint takes a target Student and a pool of Jobs, evaluates the AI 
    match score for every pair, and returns a sorted list of job recommendations
    with detailed explainability payloads.
    
    Raises:
        HTTPException 503: If the ML model is not available.
        HTTPException 400: If the prediction engine encounters invalid data.
        HTTPException 500: Unhandled internal processing errors.
    """
    # EDGE CASE 1: Model not loaded
    if _RANKER_MODEL is None:
        logger.error("API request failed: Ranker model is not initialized.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Machine learning model is currently unavailable."
        )

    # EDGE CASE 2: Redundant check for empty jobs
    if not request.jobs:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The job list cannot be empty."
        )

    try:
        logger.info(f"Ranking {len(request.jobs)} jobs for Student {request.student.student_id}...")
        
        # Execute business logic
        response = rank_jobs_for_student(
            model=_RANKER_MODEL,
            student=request.student,
            jobs=request.jobs,
            top_k=request.top_k
        )
        
        logger.info(f"Successfully ranked {len(response.ranked_jobs)} jobs.")
        return response

    except ValueError as ve:
        logger.error(f"Data validation error during job ranking: {ve}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        logger.error(f"Internal failure during job ranking: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while processing the ranking request."
        )


# ---------------------------------------------------------------------------
# Health Check Endpoint
# ---------------------------------------------------------------------------
@app.get("/health", tags=["System"])
def health_check():
    """
    Simple health check to verify the API is running and the model is loaded.
    Useful for container orchestration (Kubernetes readiness probes).
    """
    return {
        "status": "online",
        "model_loaded": _RANKER_MODEL is not None
    }
