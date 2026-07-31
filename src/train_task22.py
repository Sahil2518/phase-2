"""
train_task22.py — Data-Subject Rights & Resilience (Drift Monitoring & Retraining)

This script implements a drift monitoring pipeline. It simulates a macro-economic
data drift (e.g., skill overlap decreases, experience gaps widen). It evaluates the
current production model against this new distribution. If performance degrades 
beyond a threshold (NDCG@5 drops > 5%), it automatically retrains a new model on 
the drifted data and serializes it as v3.

Standing Instructions applied:
- NumPy style docstrings
- Defensive error handling
- Robust structured logging
- random_state=42 reproducibility
"""

import os
import sys
import json
import logging
import datetime
import hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# ---------------------------------------------------------------------------
# Logging & Setup
# ---------------------------------------------------------------------------
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/task22.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

from src.data_generator import FEATURE_COLS, _conversion_label
from src.train_task06 import _compute_ndcg, _compute_mrr
from src.train_task07 import train_tuned_ranker

# ---------------------------------------------------------------------------
# Drift Simulation
# ---------------------------------------------------------------------------
def generate_drifted_data(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic data that simulates a macro-economic drift.
    Specifically, candidates have worse skill overlap and wider experience gaps.

    Parameters
    ----------
    n_samples : int
        Number of synthetic records to create.
    random_state : int
        NumPy random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with drifted feature distributions and true labels.
    """
    try:
        rng = np.random.default_rng(random_state)
        logger.info(f"Generating {n_samples} DRIFTED records (seed={random_state})...")

        # DRIFT: Skill overlap is generally lower
        skill_overlap = rng.beta(2, 5, n_samples) 
        
        # DRIFT: Experience gap is wider (more candidates missing experience)
        raw_exp_gap = rng.uniform(-4.0, 4.0, n_samples)
        norm_exp_gap = np.clip((raw_exp_gap + 4.0) / 8.0, 0.0, 1.0)
        
        # Normal distributions for others
        raw_salary = rng.uniform(0.3, 2.0, n_samples)
        capped_salary = np.clip(raw_salary, 0.0, 1.0)
        
        edu_met = rng.integers(0, 2, n_samples).astype(float)
        coding_met = rng.integers(0, 2, n_samples).astype(float)
        comm_met = rng.integers(0, 2, n_samples).astype(float)
        loc_match = rng.integers(0, 2, n_samples).astype(float)

        df = pd.DataFrame({
            "skill_overlap_ratio":         skill_overlap,
            "norm_experience_gap":         norm_exp_gap,
            "capped_salary_ratio":         capped_salary,
            "education_met":               edu_met,
            "coding_threshold_met":        coding_met,
            "communication_threshold_met": comm_met,
            "location_match":              loc_match,
        })

        # Calculate true labels based on the actual conversion weight function
        df["label"] = df.apply(_conversion_label, axis=1)
        
        assert not df.isnull().any().any(), "NaN values found in drifted data."
        return df
    except Exception as e:
        logger.error(f"Failed to generate drifted data: {e}", exc_info=True)
        raise

# ---------------------------------------------------------------------------
# Core Pipeline
# ---------------------------------------------------------------------------
def load_latest_model(version_prefix: str = "ranker_v2_"):
    """Loads the most recent model matching the version prefix."""
    pkl_files = [f for f in os.listdir("models") if f.startswith(version_prefix) and f.endswith(".pkl")]
    if not pkl_files:
        raise FileNotFoundError(f"No model found with prefix {version_prefix} in models/.")
    pkl_files.sort(reverse=True)
    model_path = os.path.join("models", pkl_files[0])
    logger.info(f"Loading model for monitoring: {model_path}")
    return joblib.load(model_path)


def run_drift_pipeline():
    """
    Executes the drift monitoring and automated retraining pipeline.
    """
    logger.info("=" * 70)
    logger.info("Starting Task 22: Drift Monitoring & Retraining Pipeline")
    logger.info("=" * 70)

    # 1. Load baseline metrics & current model
    try:
        current_model = load_latest_model("ranker_v2_")
    except FileNotFoundError:
        # Fallback to general ranker if v2 isn't available
        current_model = load_latest_model("ranker_")
        
    if current_model is None:
        raise ValueError("Loaded model is None.")

    # 2. Baseline performance (what it used to be)
    baseline_path = "logs/task07_tuning_report.json"
    if os.path.exists(baseline_path):
        with open(baseline_path, "r") as f:
            report = json.load(f)
            baseline_ndcg = report.get("metrics_delta", {}).get("ndcg_at_5", {}).get("v2", 0.85)
    else:
        baseline_ndcg = 0.85  # Fallback assumption
        
    logger.info(f"[Monitor] Baseline NDCG@5 expected: {baseline_ndcg:.4f}")

    # 3. Evaluate on Drifted Data
    logger.info("[Monitor] Generating drifted evaluation dataset...")
    drift_eval_df = generate_drifted_data(n_samples=1000, random_state=42)
    X_eval = drift_eval_df[FEATURE_COLS]
    y_true = drift_eval_df["label"].values

    logger.info("[Monitor] Scoring drifted data with current model...")
    raw_preds = current_model.predict(X_eval)
    invalid_mask = np.isnan(raw_preds) | np.isinf(raw_preds)
    if invalid_mask.sum() > 0:
        raw_preds[invalid_mask] = 0.0
    y_pred = np.clip(raw_preds, 0.0, 1.0)

    drifted_ndcg = _compute_ndcg(y_true, y_pred, k=5)
    drifted_rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    
    logger.info(f"[Monitor] Drifted Data NDCG@5: {drifted_ndcg:.4f} (Drop: {baseline_ndcg - drifted_ndcg:.4f})")

    # 4. Drift Decision
    DRIFT_THRESHOLD = 0.05
    is_drifted = (baseline_ndcg - drifted_ndcg) >= DRIFT_THRESHOLD
    
    report_data = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "baseline_ndcg": baseline_ndcg,
        "drifted_ndcg": drifted_ndcg,
        "drift_detected": is_drifted,
        "retrained": False
    }

    if is_drifted:
        logger.warning(f"🚨 CONCEPT DRIFT DETECTED! NDCG@5 dropped by {baseline_ndcg - drifted_ndcg:.4f} (Threshold: {DRIFT_THRESHOLD})")
        logger.info("Initiating Automated Retraining Pipeline...")
        
        # 5. Retrain Model
        logger.info("[Retrain] Generating new drifted training corpus...")
        train_df = generate_drifted_data(n_samples=1500, random_state=101)
        
        logger.info("[Retrain] Training new model (v3) on drifted distribution...")
        retrained_model, _ = train_tuned_ranker(train_df)
        
        # 6. Evaluate Retrained Model
        logger.info("[Retrain] Evaluating v3 model...")
        raw_preds_v3 = retrained_model.predict(X_eval)
        invalid_mask_v3 = np.isnan(raw_preds_v3) | np.isinf(raw_preds_v3)
        if invalid_mask_v3.sum() > 0:
            raw_preds_v3[invalid_mask_v3] = 0.0
        y_pred_v3 = np.clip(raw_preds_v3, 0.0, 1.0)
        
        recovered_ndcg = _compute_ndcg(y_true, y_pred_v3, k=5)
        logger.info(f"[Retrain] Recovered NDCG@5: {recovered_ndcg:.4f} (Gain: {recovered_ndcg - drifted_ndcg:.4f})")
        
        # 7. Serialize v3 Model
        date_str = datetime.date.today().strftime("%Y%m%d")
        pkl_path = os.path.join("models", f"ranker_v3_{date_str}.pkl")
        joblib.dump(retrained_model, pkl_path)
        logger.info(f"[Retrain] Saved new model to {pkl_path}")
        
        report_data["retrained"] = True
        report_data["recovered_ndcg"] = recovered_ndcg
        report_data["new_model_artifact"] = pkl_path
    else:
        logger.info("✅ No significant drift detected. Retraining bypassed.")

    # 8. Save Report
    report_path = "logs/task22_drift_report.json"
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=4)
    logger.info(f"Drift report saved to {report_path}")
    logger.info("--- Task 22 Pipeline Complete ---")

def main():
    try:
        run_drift_pipeline()
    except Exception as e:
        logger.critical(f"Unhandled fatal error in drift pipeline: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
