"""
train_task06.py — Task 06: AI/ML Monetization — Match-Quality Baseline

Records a verified, pre-monetization quality snapshot of the existing AI
ranker. This baseline JSON is the immutable control measurement against
which all future monetization experiments (premium boosts, payment tiers)
will be compared.

Pipeline:
  1. Load existing trained ranker (models/ranker_v1_*.pkl) — NO retraining.
  2. Generate a fixed evaluation corpus (n=1000, seed=42).
  3. Score all pairs and compute ground-truth labels.
  4. Compute regression metrics (RMSE, MAE, R², Spearman ρ).
  5. Compute ranking metrics (NDCG@5, NDCG@10, MRR, Precision@5, Precision@10).
  6. Compute score distribution statistics.
  7. Extract feature importance from the model.
  8. Save logs/task06_baseline.json — the immutable pre-monetization record.
  9. Save logs/task06_baseline_chart.png — dual-panel visualisation.
 10. Log everything to logs/task06.log.

Standing instructions: robust error handling, structured logging, random_state=42.
"""

import os
import sys
import io
import json
import logging
import datetime
import hashlib

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, skew, kurtosis
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Logging — must be configured before importing project modules
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
        logging.FileHandler("logs/task06.log", encoding="utf-8"),
        _stream_handler,
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from src.data_generator import generate_synthetic_data, FEATURE_COLS, _reference_label
from src.ranker import load_ranker, train_ranker, save_ranker

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EVAL_N_SAMPLES   = 1000
EVAL_RANDOM_SEED = 42
BASELINE_VERSION = "v1.0"
SHORTLIST_THRESHOLD   = 0.65   # score >= this → shortlisted
HIGH_CONFIDENCE_SCORE = 0.70
LOW_CONFIDENCE_SCORE  = 0.40

FEATURE_LABELS = {
    "skill_overlap_ratio":          "Skill Match",
    "norm_experience_gap":          "Experience Fit",
    "capped_salary_ratio":          "Salary Alignment",
    "education_met":                "Education Req.",
    "coding_threshold_met":         "Coding Prof.",
    "communication_threshold_met":  "Communication Prof.",
    "location_match":               "Location Compat.",
}


# ---------------------------------------------------------------------------
# Step helpers
# ---------------------------------------------------------------------------

def _locate_or_train_model() -> object:
    """
    Locate the most-recent ranker .pkl in models/ and load it.

    If no pre-trained model exists (first-run scenario), trains a new one
    on the standard synthetic corpus (n=600, seed=42) so the pipeline can
    always complete without manual intervention.

    Returns
    -------
    model : fitted estimator
    """
    pkl_files = sorted(
        [f for f in os.listdir("models") if f.startswith("ranker_v1_") and f.endswith(".pkl")],
        reverse=True,
    )

    if pkl_files:
        pkl_path = os.path.join("models", pkl_files[0])
        logger.info(f"  -> Found existing model: {pkl_path}")
        return load_ranker(pkl_path), pkl_path

    logger.warning("  -> No existing model found. Training a new ranker for baseline...")
    train_df = generate_synthetic_data(n_samples=600, random_state=42)
    model, metrics = train_ranker(train_df)
    pkl_path = save_ranker(model, metrics)
    return load_ranker(pkl_path), pkl_path


def _compute_ndcg(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """
    Compute Normalised Discounted Cumulative Gain at rank k.

    Uses the sorted ground-truth scores as the ideal ranking and compares
    the predicted ranking order against it.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth relevance scores.
    y_pred : np.ndarray
        Predicted relevance scores.
    k : int
        Cutoff rank.

    Returns
    -------
    float : NDCG@k value in [0, 1].
    """
    k = min(k, len(y_true))

    # Predicted ranking order → pick top-k by predicted score
    pred_order  = np.argsort(y_pred)[::-1][:k]
    ideal_order = np.argsort(y_true)[::-1][:k]

    def dcg(scores):
        return sum(
            (2 ** scores[i] - 1) / np.log2(i + 2)
            for i in range(len(scores))
        )

    actual_gains = y_true[pred_order]
    ideal_gains  = y_true[ideal_order]

    idcg = dcg(ideal_gains)
    if idcg == 0:
        return 0.0
    return dcg(actual_gains) / idcg


def _compute_mrr(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    """
    Compute Mean Reciprocal Rank.

    A "relevant" result is any pair whose ground-truth label >= threshold.
    We evaluate across multiple simulated query groups (every 10 items).

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth relevance scores.
    y_pred : np.ndarray
        Predicted relevance scores.
    threshold : float
        Relevance threshold for a positive label.

    Returns
    -------
    float : MRR value in [0, 1].
    """
    n = len(y_true)
    group_size = 10
    reciprocal_ranks = []

    for start in range(0, n, group_size):
        end = min(start + group_size, n)
        gt  = y_true[start:end]
        pr  = y_pred[start:end]

        order   = np.argsort(pr)[::-1]
        gt_ord  = gt[order]
        relevant = np.where(gt_ord >= threshold)[0]
        if len(relevant) > 0:
            reciprocal_ranks.append(1.0 / (relevant[0] + 1))

    return float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0


def _compute_precision_at_k(
    y_true: np.ndarray, y_pred: np.ndarray, k: int, threshold: float = 0.5
) -> float:
    """
    Compute Precision@k — fraction of top-k predicted items that are truly relevant.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth relevance scores.
    y_pred : np.ndarray
        Predicted relevance scores.
    k : int
        Cutoff rank.
    threshold : float
        Relevance threshold.

    Returns
    -------
    float : Precision@k in [0, 1].
    """
    k = min(k, len(y_true))
    top_k_indices = np.argsort(y_pred)[::-1][:k]
    n_relevant = sum(1 for i in top_k_indices if y_true[i] >= threshold)
    return n_relevant / k


def _extract_feature_importance(model) -> list:
    """
    Extract per-feature importance scores from the trained model.

    Supports LightGBM, GradientBoostingRegressor, and RandomForestRegressor.
    Falls back to uniform importance if the model does not expose feature
    importances.

    Parameters
    ----------
    model : fitted estimator
        The trained ranker model.

    Returns
    -------
    list of dict : [{feature, label, importance, rank}, ...]
        Sorted descending by importance.
    """
    try:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        else:
            logger.warning("Model has no feature_importances_. Using uniform importance.")
            importances = np.ones(len(FEATURE_COLS)) / len(FEATURE_COLS)

        total = importances.sum()
        norm  = (importances / total).tolist() if total > 0 else [1.0 / len(FEATURE_COLS)] * len(FEATURE_COLS)

        result = [
            {
                "rank": 0,
                "feature": FEATURE_COLS[i],
                "label": FEATURE_LABELS.get(FEATURE_COLS[i], FEATURE_COLS[i]),
                "importance": round(float(norm[i]), 6),
            }
            for i in range(len(FEATURE_COLS))
        ]
        result.sort(key=lambda x: x["importance"], reverse=True)
        for idx, item in enumerate(result):
            item["rank"] = idx + 1

        return result

    except Exception as e:
        logger.warning(f"Feature importance extraction failed (non-fatal): {e}")
        return []


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def generate_baseline_chart(scores: np.ndarray, feature_importance: list, output_path: str):
    """
    Generate a dual-panel baseline chart:
      Panel 1 — Score distribution histogram + KDE overlay.
      Panel 2 — Feature importance horizontal bar chart.

    Parameters
    ----------
    scores : np.ndarray
        Array of predicted match scores.
    feature_importance : list
        Output of _extract_feature_importance().
    output_path : str
        File path to save the PNG.
    """
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(
            "Task 06 — Pre-Monetization Match-Quality Baseline\n"
            f"(n={len(scores):,} pairs | seed={EVAL_RANDOM_SEED} | {datetime.date.today()})",
            fontsize=13, fontweight="bold", y=1.01,
        )

        # ── Panel 1: Score Distribution ──────────────────────────────────
        ax1.hist(scores, bins=40, color="#3498db", edgecolor="#1a6496",
                 linewidth=0.5, alpha=0.85, density=True, label="Score density")

        # KDE via simple Gaussian smoothing
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(scores, bw_method=0.15)
        xs  = np.linspace(0, 1, 300)
        ax1.plot(xs, kde(xs), color="#e74c3c", linewidth=2.0, label="KDE")

        # Shortlist threshold line
        ax1.axvline(SHORTLIST_THRESHOLD, color="#f39c12", linewidth=1.8,
                    linestyle="--", label=f"Shortlist threshold ({SHORTLIST_THRESHOLD})")

        # Percentile annotations
        for pct, lbl in [(25, "p25"), (50, "p50"), (75, "p75")]:
            val = float(np.percentile(scores, pct))
            ax1.axvline(val, color="#95a5a6", linewidth=1.0, linestyle=":")
            ax1.text(val + 0.01, ax1.get_ylim()[1] * 0.9, lbl, fontsize=8, color="#666")

        ax1.set_xlabel("Match Score", fontsize=11)
        ax1.set_ylabel("Density", fontsize=11)
        ax1.set_title("Score Distribution", fontsize=12, fontweight="bold")
        ax1.legend(fontsize=9)
        ax1.set_xlim(0, 1)

        # ── Panel 2: Feature Importance ──────────────────────────────────
        if feature_importance:
            labels = [fi["label"] for fi in feature_importance]
            values = [fi["importance"] for fi in feature_importance]
            colors = [
                "#2ecc71" if v >= 0.20 else "#f39c12" if v >= 0.10 else "#e74c3c"
                for v in values
            ]
            y_pos = np.arange(len(labels))

            bars = ax2.barh(y_pos, values, color=colors, edgecolor="#333", linewidth=0.5)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(labels, fontsize=10)
            ax2.invert_yaxis()
            ax2.set_xlabel("Normalised Feature Importance", fontsize=11)
            ax2.set_title("Feature Importance Ranking", fontsize=12, fontweight="bold")

            for bar, val in zip(bars, values):
                ax2.text(
                    bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1%}", va="center", fontsize=9, fontweight="bold",
                )

            legend_patches = [
                mpatches.Patch(color="#2ecc71", label="High (>=20%)"),
                mpatches.Patch(color="#f39c12", label="Moderate (>=10%)"),
                mpatches.Patch(color="#e74c3c", label="Low (<10%)"),
            ]
            ax2.legend(handles=legend_patches, loc="lower right", fontsize=9)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"  -> Baseline chart saved to {output_path}")

    except Exception as e:
        logger.error(f"Baseline chart generation failed (non-fatal): {e}", exc_info=True)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline():
    """
    Execute the full Task 06 match-quality baseline recording pipeline.

    Steps
    -----
    1. Load the existing trained ranker — no retraining.
    2. Generate fixed evaluation corpus (n=1000, seed=42).
    3. Score all pairs using the model.
    4. Compute regression metrics.
    5. Compute ranking quality metrics (NDCG, MRR, Precision@K).
    6. Compute score distribution statistics.
    7. Extract feature importance.
    8. Assemble and write the baseline JSON.
    9. Generate and save the dual-panel baseline chart.
    """
    logger.info("=" * 70)
    logger.info("Starting Task 06 Pipeline: Match-Quality Baseline Recording")
    logger.info(f"  Baseline version : {BASELINE_VERSION}")
    logger.info(f"  Evaluation corpus: n={EVAL_N_SAMPLES}, seed={EVAL_RANDOM_SEED}")
    logger.info("=" * 70)

    # ── Step 1: Load model ────────────────────────────────────────────────
    logger.info("[Step 1/9] Loading trained ranker model...")
    model, pkl_path = _locate_or_train_model()

    if model is None:
        raise ValueError("Cannot record baseline: model is uninitialized or None.")

    # Compute SHA-256 of model artifact for traceability
    with open(pkl_path, "rb") as f:
        model_sha256 = hashlib.sha256(f.read()).hexdigest()
    logger.info(f"  -> Model SHA-256: {model_sha256[:16]}...")

    # ── Step 2: Generate evaluation corpus ───────────────────────────────
    logger.info("[Step 2/9] Generating fixed evaluation corpus...")
    eval_df = generate_synthetic_data(n_samples=EVAL_N_SAMPLES, random_state=EVAL_RANDOM_SEED)

    if eval_df.empty:
        raise ValueError("Evaluation corpus is empty — cannot record baseline.")

    y_true = eval_df["label"].values
    X_eval = eval_df[FEATURE_COLS]
    logger.info(f"  -> Corpus: {len(eval_df)} pairs, {len(FEATURE_COLS)} features each.")

    # ── Step 3: Score all pairs ───────────────────────────────────────────
    logger.info("[Step 3/9] Scoring evaluation corpus with the ranker model...")

    # Guard: model must not be None
    if model is None:
        raise ValueError("Model is None — cannot score pairs.")

    raw_preds = model.predict(X_eval)

    # Guard: NaN / Inf output
    invalid_mask = np.isnan(raw_preds) | np.isinf(raw_preds)
    n_invalid = int(invalid_mask.sum())
    if n_invalid > 0:
        logger.warning(f"  -> {n_invalid} invalid model outputs detected. Clamping to 0.0.")
        raw_preds[invalid_mask] = 0.0

    y_pred = np.clip(raw_preds, 0.0, 1.0)
    logger.info(f"  -> Scoring complete. Score range: [{y_pred.min():.4f}, {y_pred.max():.4f}]")

    # ── Step 4: Regression metrics ────────────────────────────────────────
    logger.info("[Step 4/9] Computing regression metrics...")
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    rho, pval = spearmanr(y_true, y_pred)
    spearman_rho  = float(rho)
    spearman_pval = float(pval)

    logger.info(f"  -> RMSE={rmse:.6f} | MAE={mae:.6f} | R²={r2:.6f} | Spearman ρ={spearman_rho:.6f}")

    regression_metrics = {
        "rmse":          round(rmse, 6),
        "mae":           round(mae, 6),
        "r2":            round(r2, 6),
        "spearman_rho":  round(spearman_rho, 6),
        "spearman_pval": round(spearman_pval, 6),
    }

    # ── Step 5: Ranking quality metrics ──────────────────────────────────
    logger.info("[Step 5/9] Computing ranking quality metrics (NDCG, MRR, Precision@K)...")
    ndcg_5   = _compute_ndcg(y_true, y_pred, k=5)
    ndcg_10  = _compute_ndcg(y_true, y_pred, k=10)
    mrr      = _compute_mrr(y_true, y_pred, threshold=0.5)
    prec_5   = _compute_precision_at_k(y_true, y_pred, k=5)
    prec_10  = _compute_precision_at_k(y_true, y_pred, k=10)

    logger.info(
        f"  -> NDCG@5={ndcg_5:.4f} | NDCG@10={ndcg_10:.4f} | "
        f"MRR={mrr:.4f} | P@5={prec_5:.4f} | P@10={prec_10:.4f}"
    )

    ranking_metrics = {
        "ndcg_at_5":      round(ndcg_5, 6),
        "ndcg_at_10":     round(ndcg_10, 6),
        "mrr":            round(mrr, 6),
        "precision_at_5": round(prec_5, 6),
        "precision_at_10":round(prec_10, 6),
    }

    # ── Step 6: Score distribution statistics ────────────────────────────
    logger.info("[Step 6/9] Computing score distribution statistics...")
    score_skew = float(skew(y_pred))
    score_kurt = float(kurtosis(y_pred))

    dist_stats = {
        "mean":     round(float(np.mean(y_pred)), 6),
        "std":      round(float(np.std(y_pred)), 6),
        "min":      round(float(np.min(y_pred)), 6),
        "max":      round(float(np.max(y_pred)), 6),
        "p25":      round(float(np.percentile(y_pred, 25)), 6),
        "p50":      round(float(np.percentile(y_pred, 50)), 6),
        "p75":      round(float(np.percentile(y_pred, 75)), 6),
        "p90":      round(float(np.percentile(y_pred, 90)), 6),
        "skewness": round(score_skew, 6),
        "kurtosis": round(score_kurt, 6),
    }
    logger.info(
        f"  -> mean={dist_stats['mean']:.4f} | std={dist_stats['std']:.4f} | "
        f"p50={dist_stats['p50']:.4f} | skew={score_skew:.4f}"
    )

    # ── Step 7: Shortlist statistics ──────────────────────────────────────
    logger.info("[Step 7/9] Computing shortlist and confidence statistics...")
    n_shortlisted = int((y_pred >= SHORTLIST_THRESHOLD).sum())
    n_high   = int((y_pred >= HIGH_CONFIDENCE_SCORE).sum())
    n_low    = int((y_pred <  LOW_CONFIDENCE_SCORE).sum())
    n_medium = EVAL_N_SAMPLES - n_high - n_low

    shortlist_stats = {
        "shortlist_threshold":    SHORTLIST_THRESHOLD,
        "shortlisted_count":      n_shortlisted,
        "shortlist_rate":         round(n_shortlisted / EVAL_N_SAMPLES, 6),
        "confidence_distribution":{
            "high":   n_high,
            "medium": n_medium,
            "low":    n_low,
        },
    }
    logger.info(
        f"  -> Shortlisted: {n_shortlisted}/{EVAL_N_SAMPLES} "
        f"({shortlist_stats['shortlist_rate']:.1%}) | "
        f"High: {n_high} | Med: {n_medium} | Low: {n_low}"
    )

    # ── Step 8: Feature importance ────────────────────────────────────────
    logger.info("[Step 8/9] Extracting feature importance from model...")
    feature_importance = _extract_feature_importance(model)
    if feature_importance:
        logger.info("  -> Top-3 features by importance:")
        for fi in feature_importance[:3]:
            logger.info(f"       #{fi['rank']} {fi['label']}: {fi['importance']:.1%}")

    # ── Assemble & save baseline JSON ────────────────────────────────────
    logger.info("[Step 9/9] Saving baseline JSON and chart...")
    timestamp = datetime.datetime.utcnow().isoformat() + "Z"

    baseline = {
        "task":              "Task 06 — Match-Quality Baseline",
        "baseline_version":  BASELINE_VERSION,
        "recorded_at":       timestamp,
        "corpus": {
            "n_pairs":      EVAL_N_SAMPLES,
            "random_seed":  EVAL_RANDOM_SEED,
            "features":     FEATURE_COLS,
        },
        "model": {
            "artifact":     pkl_path,
            "sha256":       model_sha256,
        },
        "regression_metrics":   regression_metrics,
        "ranking_metrics":      ranking_metrics,
        "score_distribution":   dist_stats,
        "shortlist_stats":      shortlist_stats,
        "feature_importance":   feature_importance,
    }

    baseline_path = "logs/task06_baseline.json"
    with open(baseline_path, "w", encoding="utf-8") as f:
        json.dump(baseline, f, indent=2, ensure_ascii=False)
    logger.info(f"  -> Baseline JSON saved to {baseline_path}")

    # Generate chart
    chart_path = "logs/task06_baseline_chart.png"
    generate_baseline_chart(y_pred, feature_importance, chart_path)

    # Console summary
    print(f"\n{'=' * 65}")
    print(f"  TASK 06 -- MATCH-QUALITY BASELINE RECORDED")
    print(f"{'=' * 65}")
    print(f"  Model artifact    : {pkl_path}")
    print(f"  Evaluation pairs  : {EVAL_N_SAMPLES:,}  (seed={EVAL_RANDOM_SEED})")
    print(f"\n  -- Regression --")
    print(f"  RMSE              : {rmse:.6f}")
    print(f"  MAE               : {mae:.6f}")
    print(f"  R2                : {r2:.6f}")
    print(f"  Spearman rho      : {spearman_rho:.6f}  (p={spearman_pval:.4g})")
    print(f"\n  -- Ranking --")
    print(f"  NDCG@5            : {ndcg_5:.4f}")
    print(f"  NDCG@10           : {ndcg_10:.4f}")
    print(f"  MRR               : {mrr:.4f}")
    print(f"  Precision@5       : {prec_5:.4f}")
    print(f"  Precision@10      : {prec_10:.4f}")
    print(f"\n  -- Distribution --")
    print(f"  Mean score        : {dist_stats['mean']:.4f}  +-  {dist_stats['std']:.4f}")
    print(f"  p50 (median)      : {dist_stats['p50']:.4f}")
    print(f"  Shortlist rate    : {shortlist_stats['shortlist_rate']:.1%}")
    print(f"\n  -- Outputs --")
    print(f"  Baseline JSON     : {baseline_path}")
    print(f"  Chart             : {chart_path}")
    print(f"  Log               : logs/task06.log")
    print(f"{'=' * 65}\n")

    logger.info("=" * 70)
    logger.info("Task 06 Pipeline completed successfully.")
    logger.info(f"  Baseline JSON : {baseline_path}")
    logger.info(f"  Chart         : {chart_path}")
    logger.info("=" * 70)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        run_pipeline()
    except FileNotFoundError as e:
        logger.critical(f"Missing required file: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Unhandled fatal error: {e}", exc_info=True)
        sys.exit(1)
