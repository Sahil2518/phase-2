"""
ranker.py — AI-Powered Ranking Engine for Task 03

Trains a LightGBM (or Random Forest fallback) regressor on synthetic
(student, job) relevance labels and uses it to rank:
  • Jobs for a given student
  • Candidates for a given job

Standing instructions: robust error handling, structured logging.
"""

import os
import sys
import json
import hashlib
import datetime
import logging
import joblib
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr

from src.data_generator import FEATURE_COLS, match_vector_to_feature_row
from src.match_vectors import compute_match_vector
from src.model_schemas import (
    StudentFeatures, JobFeatures, MatchVector,
    RankedResult, StudentRankingResponse, JobRankingResponse,
    ExplanationPayload,
)
from src.explainer import build_explanation_payload

# ---------------------------------------------------------------------------
# Logging (handlers already set up by data_generator / train_task03)
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model persistence paths
# ---------------------------------------------------------------------------
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Explanation helper
# ---------------------------------------------------------------------------
FEATURE_LABELS = {
    "skill_overlap_ratio":          "Skill match",
    "norm_experience_gap":          "Experience fit",
    "capped_salary_ratio":          "Salary alignment",
    "education_met":                "Education requirement met",
    "coding_threshold_met":         "Coding threshold met",
    "communication_threshold_met":  "Communication threshold met",
    "location_match":               "Location / remote preference match",
}

def _build_explanation(feature_row: dict, score: float) -> List[str]:
    """Generate human-readable positive/negative factors for a ranked result."""
    explanations = []
    if feature_row["skill_overlap_ratio"] >= 0.7:
        explanations.append(f"✅ Strong skill match ({feature_row['skill_overlap_ratio']:.0%})")
    elif feature_row["skill_overlap_ratio"] >= 0.4:
        explanations.append(f"⚠️  Partial skill match ({feature_row['skill_overlap_ratio']:.0%})")
    else:
        explanations.append(f"❌ Weak skill match ({feature_row['skill_overlap_ratio']:.0%})")

    if feature_row["norm_experience_gap"] >= 0.6:
        explanations.append("✅ Experience well within range")
    elif feature_row["norm_experience_gap"] >= 0.4:
        explanations.append("⚠️  Experience near the minimum threshold")
    else:
        explanations.append("❌ Experience below requirement")

    if feature_row["education_met"]:
        explanations.append("✅ Education requirement satisfied")
    else:
        explanations.append("❌ Education requirement not met")

    if feature_row["coding_threshold_met"]:
        explanations.append("✅ Coding score above threshold")
    else:
        explanations.append("❌ Coding score below threshold")

    if feature_row["communication_threshold_met"]:
        explanations.append("✅ Communication score above threshold")
    else:
        explanations.append("❌ Communication score below threshold")

    if feature_row["location_match"]:
        explanations.append("✅ Location / work model compatible")
    else:
        explanations.append("⚠️  Location mismatch")

    return explanations


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_ranker(df: pd.DataFrame) -> Tuple[Any, Dict[str, float]]:
    """
    Train a gradient-boosted regressor on (features, label) pairs.

    Tries LightGBM first; falls back to sklearn GradientBoostingRegressor,
    then RandomForestRegressor.

    Returns
    -------
    model : fitted estimator
    metrics : dict with rmse, r2, spearman_rho
    """
    try:
        assert len(df) > 0, "Training DataFrame is empty."
        assert "label" in df.columns, "'label' column missing."
        for col in FEATURE_COLS:
            assert col in df.columns, f"Feature column '{col}' missing."

        X = df[FEATURE_COLS]  # Keep as DataFrame so feature names are preserved
        y = df["label"].values

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.20, random_state=42
        )
        logger.info(f"Training set: {len(X_train)} rows | Validation set: {len(X_val)} rows")

        # --- attempt LightGBM ---
        model = None
        model_name = ""
        try:
            import lightgbm as lgb
            model = lgb.LGBMRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1,
            )
            model_name = "LightGBM"
            logger.info("LightGBM available — using LGBMRegressor.")
        except ImportError:
            logger.warning("LightGBM not installed. Falling back to GradientBoostingRegressor.")
            model = GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=5,
                random_state=42,
            )
            model_name = "GradientBoostingRegressor"

        model.fit(X_train, y_train)
        logger.info(f"{model_name} training complete.")

        # --- evaluation ---
        y_pred = model.predict(X_val)
        rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
        r2   = float(r2_score(y_val, y_pred))
        rho, pval = spearmanr(y_val, y_pred)
        spearman_rho = float(rho)

        metrics = {
            "model": model_name,
            "n_train": int(len(X_train)),
            "n_val":   int(len(X_val)),
            "rmse":    round(rmse, 6),
            "r2":      round(r2, 6),
            "spearman_rho": round(spearman_rho, 6),
            "spearman_pval": round(float(pval), 6),
        }

        logger.info(
            f"Validation — RMSE: {rmse:.4f} | R²: {r2:.4f} | "
            f"Spearman ρ: {spearman_rho:.4f} (p={pval:.4g})"
        )
        return model, metrics

    except Exception as e:
        logger.error(f"Model training failed: {e}", exc_info=True)
        raise


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------
def save_ranker(model: Any, metrics: Dict[str, float]) -> str:
    """
    Serialise the trained model with joblib.
    Writes .pkl, .sha256, and _metadata.json sidecar.

    Returns
    -------
    str : path to the .pkl file
    """
    try:
        date_str = datetime.date.today().strftime("%Y%m%d")
        base_name = f"ranker_v1_{date_str}"
        pkl_path  = os.path.join(MODELS_DIR, f"{base_name}.pkl")
        sha_path  = os.path.join(MODELS_DIR, f"{base_name}.sha256")
        meta_path = os.path.join(MODELS_DIR, f"{base_name}_metadata.json")

        joblib.dump(model, pkl_path)
        logger.info(f"Model serialised → {pkl_path}")

        # SHA-256 integrity check
        with open(pkl_path, "rb") as f:
            checksum = hashlib.sha256(f.read()).hexdigest()
        with open(sha_path, "w") as f:
            f.write(checksum)
        logger.info(f"SHA-256: {checksum[:16]}...  → {sha_path}")

        # Metadata sidecar
        metadata = {
            "version":    "v1",
            "date":       date_str,
            "features":   FEATURE_COLS,
            "metrics":    metrics,
            "sha256":     checksum,
            "artifact":   pkl_path,
        }
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Metadata sidecar → {meta_path}")

        return pkl_path

    except Exception as e:
        logger.error(f"Model serialisation failed: {e}", exc_info=True)
        raise


def load_ranker(pkl_path: str) -> Any:
    """Load a serialised ranker and verify its SHA-256 checksum."""
    try:
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Model artifact not found: {pkl_path}")

        sha_path = pkl_path.replace(".pkl", ".sha256")
        if os.path.exists(sha_path):
            with open(pkl_path, "rb") as f:
                actual = hashlib.sha256(f.read()).hexdigest()
            with open(sha_path, "r") as f:
                expected = f.read().strip()
            if actual != expected:
                raise ValueError(
                    f"SHA-256 mismatch! Expected {expected[:16]}... got {actual[:16]}..."
                )
            logger.info("SHA-256 integrity check passed.")
        else:
            logger.warning("No .sha256 sidecar found — skipping integrity check.")

        model = joblib.load(pkl_path)
        logger.info(f"Model loaded from {pkl_path}")
        return model

    except Exception as e:
        logger.error(f"Model loading failed: {e}", exc_info=True)
        raise


# ---------------------------------------------------------------------------
# Scoring & Ranking
# ---------------------------------------------------------------------------
def score_pair(model: Any, mv: MatchVector) -> Tuple[float, dict]:
    """
    Predict relevance score for a (student, job) pair given a MatchVector.

    Returns
    -------
    score : float in [0, 1]
    feature_row : dict of features used
    """
    try:
        feature_row = match_vector_to_feature_row(mv)
        # Pass as DataFrame so feature names match training
        X = pd.DataFrame([feature_row], columns=FEATURE_COLS)
        raw_score = float(model.predict(X)[0])
        score = float(np.clip(raw_score, 0.0, 1.0))
        return score, feature_row
    except Exception as e:
        logger.error(f"Scoring failed for pair ({mv.student_id}, {mv.job_id}): {e}", exc_info=True)
        raise


def rank_jobs_for_student(
    model: Any,
    student: StudentFeatures,
    jobs: List[JobFeatures],
    top_k: int = None,
) -> StudentRankingResponse:
    """
    Rank all jobs for a given student.

    Returns
    -------
    StudentRankingResponse with ranked_jobs sorted descending by score.
    """
    try:
        ranked = []
        for job in jobs:
            try:
                mv = compute_match_vector(student, job)
                score, feature_row = score_pair(model, mv)
                explanation = _build_explanation(feature_row, score)
                payload = build_explanation_payload(model, feature_row, score)
                ranked.append(RankedResult(
                    id=job.job_id,
                    score=round(score, 4),
                    explanation=explanation,
                    explanation_payload=payload,
                ))
            except Exception as e:
                logger.warning(f"Skipping job {job.job_id} for student {student.student_id}: {e}")

        ranked.sort(key=lambda r: r.score, reverse=True)
        if top_k is not None:
            ranked = ranked[:top_k]

        return StudentRankingResponse(
            student_id=student.student_id,
            ranked_jobs=ranked,
            total_evaluated=len(jobs),
            timestamp=datetime.datetime.utcnow().isoformat() + "Z",
        )
    except Exception as e:
        logger.error(f"rank_jobs_for_student failed for {student.student_id}: {e}", exc_info=True)
        raise


def rank_candidates_for_job(
    model: Any,
    job: JobFeatures,
    students: List[StudentFeatures],
    top_k: int = None,
) -> JobRankingResponse:
    """
    Rank all students (candidates) for a given job.

    Returns
    -------
    JobRankingResponse with ranked_candidates sorted descending by score.
    """
    try:
        ranked = []
        for student in students:
            try:
                mv = compute_match_vector(student, job)
                score, feature_row = score_pair(model, mv)
                explanation = _build_explanation(feature_row, score)
                payload = build_explanation_payload(model, feature_row, score)
                ranked.append(RankedResult(
                    id=student.student_id,
                    score=round(score, 4),
                    explanation=explanation,
                    explanation_payload=payload,
                ))
            except Exception as e:
                logger.warning(f"Skipping student {student.student_id} for job {job.job_id}: {e}")

        ranked.sort(key=lambda r: r.score, reverse=True)
        if top_k is not None:
            ranked = ranked[:top_k]

        return JobRankingResponse(
            job_id=job.job_id,
            ranked_candidates=ranked,
            total_evaluated=len(students),
            timestamp=datetime.datetime.utcnow().isoformat() + "Z",
        )
    except Exception as e:
        logger.error(f"rank_candidates_for_job failed for {job.job_id}: {e}", exc_info=True)
        raise
