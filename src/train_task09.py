"""
train_task09.py — Task 09: Failure Handling & Resilience

Confirms the paywall hasn't skewed relevance. Evaluates the conversion-tuned ranker (v2) 
against the original Task 6 baseline labels to ensure no relevance regression.

Pipeline:
  1. Load Task 6 baseline (logs/task06_baseline.json).
  2. Generate original reference evaluation corpus (n=1000, seed=42) using baseline labels.
  3. Load the conversion-tuned v2 model (from Task 7).
  4. Score evaluation corpus and compute metrics against baseline labels.
  5. Compare metrics against Task 6 baseline metrics to detect regression.
  6. Assert NDCG@5 hasn't dropped by more than 0.05.
  7. Save logs/task09_regression_report.json.
  8. Save logs/task09_regression_chart.png.

Standing instructions: robust error handling, structured logging, defensive checks.
"""

import os
import sys
import io
import json
import logging
import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

_stream_handler = logging.StreamHandler(
    stream=io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
)
_stream_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/task09.log", encoding="utf-8"),
        _stream_handler,
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
try:
    from src.data_generator import generate_synthetic_data, FEATURE_COLS
    from src.train_task06 import (
        _compute_ndcg, _compute_mrr, _compute_precision_at_k,
        SHORTLIST_THRESHOLD
    )
    from src.ranker import load_ranker
except ImportError as e:
    logger.critical(f"Failed to import required modules: {e}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EVAL_N_SAMPLES = 1000
EVAL_RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_regression_chart(b_rank, ndcg_5_v2, mrr_v2, output_path):
    """
    Generates a chart comparing the v1 baseline ranking metrics with the v2 metrics 
    evaluated on the baseline reference dataset.

    Parameters
    ----------
    b_rank : dict
        Baseline ranking metrics dictionary.
    ndcg_5_v2 : float
        NDCG@5 score for the v2 model on the reference dataset.
    mrr_v2 : float
        MRR score for the v2 model on the reference dataset.
    output_path : str
        File path to save the chart.
    """
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.suptitle(
            "Task 09 — Relevance Regression Check (v1 vs v2 on Ref Labels)",
            fontsize=12, fontweight="bold"
        )
        
        metrics_labels = ["NDCG@5", "MRR"]
        v1_vals = [b_rank.get("ndcg_at_5", 0.0), b_rank.get("mrr", 0.0)]
        v2_vals = [ndcg_5_v2, mrr_v2]
        
        x = np.arange(len(metrics_labels))
        width = 0.35
        
        ax.bar(x - width/2, v1_vals, width, label="v1 (Original Ref)", color="#95a5a6")
        bars = ax.bar(x + width/2, v2_vals, width, label="v2 (Conversion-Tuned)", color="#e74c3c")
        
        # Draw threshold line for allowable regression (delta -0.05) on NDCG@5
        ndcg_allowable = max(0, v1_vals[0] - 0.05)
        ax.axhline(ndcg_allowable, color="red", linestyle="--", linewidth=1.5, alpha=0.5, label="NDCG Regression Threshold")
        
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_labels)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Metric Score")
        ax.legend()
        
        # Annotate bars
        for idx, rects in enumerate([ax.patches[:2], ax.patches[2:]]):
            for bar in rects:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"  -> Regression chart saved to {output_path}")

    except Exception as e:
        logger.error(f"Chart generation failed: {e}", exc_info=True)


def run_pipeline():
    """Main pipeline execution for the regression check."""
    logger.info("=" * 70)
    logger.info("Starting Task 09 Pipeline: Failure Handling & Resilience")
    logger.info("=" * 70)

    # 1. Load Baseline
    baseline_path = "logs/task06_baseline.json"
    if not os.path.exists(baseline_path):
        raise FileNotFoundError(f"Missing baseline file: {baseline_path}")
    
    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline = json.load(f)
    logger.info("[Step 1/6] Loaded Task 06 baseline (v1 model).")

    # 2. Generate original reference corpus (n=1000, seed=42)
    # Using generate_synthetic_data which yields original 'reference_label'
    logger.info("[Step 2/6] Generating reference evaluation corpus (original labels)...")
    eval_df = generate_synthetic_data(n_samples=EVAL_N_SAMPLES, random_state=EVAL_RANDOM_SEED)
    
    assert len(eval_df) > 0, "Evaluation dataframe is empty!"
    
    X_eval = eval_df[FEATURE_COLS]
    y_true_ref = eval_df["label"].values

    # 3. Load Conversion Model (v2)
    logger.info("[Step 3/6] Loading conversion-tuned v2 model...")
    pkl_files = sorted(
        [f for f in os.listdir("models") if f.startswith("ranker_v2_") and f.endswith(".pkl")],
        reverse=True
    )
    if not pkl_files:
        raise FileNotFoundError("No v2 model found in models/ directory. Run Task 07 first.")
    
    model_path = os.path.join("models", pkl_files[0])
    model_v2 = load_ranker(model_path)
    
    if model_v2 is None:
        raise ValueError(f"Failed to load model from {model_path}.")

    # 4. Evaluate
    logger.info("[Step 4/6] Scoring reference corpus with v2 model...")
    raw_preds_v2 = model_v2.predict(X_eval)
    
    # NaN/Inf guard
    invalid_mask = np.isnan(raw_preds_v2) | np.isinf(raw_preds_v2)
    if invalid_mask.sum() > 0:
        logger.warning(f"  -> {invalid_mask.sum()} invalid v2 predictions. Clamping to 0.0.")
        raw_preds_v2[invalid_mask] = 0.0
    
    y_pred_v2 = np.clip(raw_preds_v2, 0.0, 1.0)
    
    # Metrics
    rmse_v2 = float(np.sqrt(mean_squared_error(y_true_ref, y_pred_v2)))
    r2_v2   = float(r2_score(y_true_ref, y_pred_v2))

    ndcg_5_v2  = _compute_ndcg(y_true_ref, y_pred_v2, k=5)
    mrr_v2     = _compute_mrr(y_true_ref, y_pred_v2, threshold=0.5)

    n_shortlisted_v2 = int((y_pred_v2 >= SHORTLIST_THRESHOLD).sum())
    shortlist_rate_v2 = n_shortlisted_v2 / EVAL_N_SAMPLES

    # 5. Reporting
    logger.info("[Step 5/6] Generating regression report and chart...")
    b_rank = baseline.get("ranking_metrics", {})
    
    delta_report = {
        "task": "Task 09 — Relevance Regression Check",
        "recorded_at": datetime.datetime.utcnow().isoformat() + "Z",
        "models": {
            "v1": baseline["model"]["artifact"],
            "v2": model_path
        },
        "evaluation_corpus": {
            "n_samples": EVAL_N_SAMPLES,
            "seed": EVAL_RANDOM_SEED,
            "label_type": "reference_label"
        },
        "metrics_delta": {
            "ndcg_at_5": {
                "v1_baseline": b_rank.get("ndcg_at_5", 0.0),
                "v2_tuned": round(ndcg_5_v2, 6),
                "diff": round(ndcg_5_v2 - b_rank.get("ndcg_at_5", 0.0), 6)
            },
            "mrr": {
                "v1_baseline": b_rank.get("mrr", 0.0),
                "v2_tuned": round(mrr_v2, 6),
                "diff": round(mrr_v2 - b_rank.get("mrr", 0.0), 6)
            }
        }
    }
    
    report_path = "logs/task09_regression_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(delta_report, f, indent=2, ensure_ascii=False)
        
    chart_path = "logs/task09_regression_chart.png"
    generate_regression_chart(b_rank, ndcg_5_v2, mrr_v2, chart_path)

    # 6. Assertions (The core of the quality check)
    logger.info("[Step 6/6] Verifying no relevance regression...")
    
    v1_ndcg = b_rank.get("ndcg_at_5", 0.0)
    ndcg_drop = v1_ndcg - ndcg_5_v2
    
    logger.info(f"NDCG@5 delta: {ndcg_5_v2 - v1_ndcg:+.4f}")
    if ndcg_drop > 0.05:
        logger.error(f"RELEVANCE REGRESSION DETECTED: NDCG@5 dropped by {ndcg_drop:.4f} (Allowed <= 0.05)")
        raise AssertionError("Relevance regression guardrail triggered! The paywall tuning skewed relevance too much.")
    
    logger.info("PASS: No significant relevance regression detected.")

    print(f"\n{'=' * 65}")
    print(f"  TASK 09 -- RELEVANCE REGRESSION CHECK COMPLETE")
    print(f"{'=' * 65}")
    print(f"  Model Evaluated   : {model_path}")
    print(f"  Target Labels     : Original Reference Labels (Baseline)")
    print(f"\n  -- Metrics Delta (v1 -> v2) --")
    print(f"  NDCG@5            : {v1_ndcg:.4f} -> {ndcg_5_v2:.4f} (Delta: {ndcg_5_v2 - v1_ndcg:+.4f})")
    print(f"  MRR               : {b_rank.get('mrr', 0.0):.4f} -> {mrr_v2:.4f} (Delta: {mrr_v2 - b_rank.get('mrr', 0.0):+.4f})")
    print(f"\n  -- Status --")
    print(f"  GUARDRAIL PASSED. No severe relevance regression.")
    print(f"\n  -- Outputs --")
    print(f"  Delta Report      : {report_path}")
    print(f"  Chart             : {chart_path}")
    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception as e:
        logger.critical(f"Unhandled fatal error: {e}", exc_info=True)
        sys.exit(1)
