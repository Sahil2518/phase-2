"""
train_task13.py - Task 13: Proctoring False-Positive Reduction (Ship)

Extends Task 11 baseline by:
  1. Richer synthetic proctoring signals (7 features vs 4)
  2. Ensemble classifier (LightGBM + Logistic Regression soft voting)
  3. Stricter FPR target (<=3% vs <=5% in Task 11)
  4. Side-by-side comparison report vs Task 11 baseline

Standing instructions: robust error handling, structured logging, random_state=42.
"""

import os
import sys
import json
import pickle
import logging
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/task13.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TARGET_FPR = 0.03          # Stricter than Task 11's 5%
MODEL_PATH  = "models/proctor_v2_task13.pkl"
METRICS_PATH = "logs/task13_metrics.json"
CHART_PATH   = "logs/task13_fp_comparison.png"

# Task 11 baseline (from logs/task11_metrics.json)
BASELINE = {
    "threshold": 0.0,
    "roc_auc":   1.0,
    "precision": 0.7955,
    "recall_tpr": 1.0,
    "false_positive_rate": 0.0474,
    "false_positives": 9,
    "true_negatives": 181,
}

FEATURE_COLS = [
    "face_match_confidence",
    "background_noise_level",
    "tab_switch_count",
    "keystroke_variance",
    # NEW in Task 13
    "gaze_deviation_score",
    "audio_mismatch_score",
    "typing_speed_consistency",
]


# ---------------------------------------------------------------------------
# Step 1 — Enhanced Data Generator
# ---------------------------------------------------------------------------
def generate_enhanced_proctoring_data(
    n_samples: int = 2000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic proctoring events with 7 signals (vs 4 in Task 11).

    Additional signals improve class separability and reduce ambiguity that
    causes false positives in the simpler 4-feature model.

    Parameters
    ----------
    n_samples : int
        Number of synthetic records.
    random_state : int
        NumPy seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Feature matrix with binary 'label' column.
    """
    try:
        rng = np.random.default_rng(random_state)
        logger.info(f"Generating {n_samples} enhanced proctoring records (seed={random_state})...")

        labels = rng.binomial(1, 0.15, n_samples)

        # --- Original 4 signals ---
        face_match = np.where(
            labels == 1,
            np.clip(rng.normal(0.60, 0.20, n_samples), 0, 1),
            np.clip(rng.normal(0.95, 0.05, n_samples), 0, 1),
        )
        noise = np.where(
            labels == 1,
            np.clip(rng.exponential(0.4, n_samples), 0, 1),
            np.clip(rng.exponential(0.1, n_samples), 0, 1),
        )
        tabs = np.where(
            labels == 1,
            np.clip(rng.poisson(8, n_samples), 0, 50),
            np.clip(rng.poisson(1, n_samples), 0, 50),
        )
        keystroke = np.where(
            labels == 1,
            np.clip(rng.normal(0.7, 0.20, n_samples), 0, 1),
            np.clip(rng.normal(0.2, 0.10, n_samples), 0, 1),
        )

        # --- 3 NEW signals (Task 13 additions) ---
        # gaze_deviation_score: how much eye gaze deviates from screen
        gaze = np.where(
            labels == 1,
            np.clip(rng.normal(0.65, 0.18, n_samples), 0, 1),
            np.clip(rng.normal(0.10, 0.08, n_samples), 0, 1),
        )
        # audio_mismatch_score: mismatch between mic audio and lip movement
        audio = np.where(
            labels == 1,
            np.clip(rng.normal(0.60, 0.20, n_samples), 0, 1),
            np.clip(rng.normal(0.05, 0.05, n_samples), 0, 1),
        )
        # typing_speed_consistency: std of inter-key intervals (high = erratic)
        typing_cons = np.where(
            labels == 1,
            np.clip(rng.normal(0.75, 0.15, n_samples), 0, 1),
            np.clip(rng.normal(0.15, 0.10, n_samples), 0, 1),
        )

        df = pd.DataFrame({
            "face_match_confidence":    face_match,
            "background_noise_level":   noise,
            "tab_switch_count":         tabs.astype(float),
            "keystroke_variance":       keystroke,
            "gaze_deviation_score":     gaze,
            "audio_mismatch_score":     audio,
            "typing_speed_consistency": typing_cons,
            "label":                    labels,
        })

        assert len(df) == n_samples
        assert df.isnull().sum().sum() == 0
        logger.info(f"Enhanced data generated. Fraud rate: {df['label'].mean():.2%}")
        return df

    except Exception as e:
        logger.critical(f"Data generation failed: {e}", exc_info=True)
        raise


# ---------------------------------------------------------------------------
# Step 2 — Train Ensemble
# ---------------------------------------------------------------------------
def train_ensemble(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> dict:
    """
    Train a soft-voting ensemble of LightGBM + Logistic Regression.

    Ensemble reduces individual model bias, which is a key source of false
    positives in single-model classifiers on imbalanced proctoring data.

    Parameters
    ----------
    X_train : pd.DataFrame
    y_train : pd.Series

    Returns
    -------
    dict with keys 'lgbm' and 'lr_pipeline'
    """
    try:
        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()
        spw = neg / pos if pos > 0 else 1.0
        logger.info(f"Class balance — Innocent: {neg}, Fraud: {pos} | scale_pos_weight: {spw:.2f}")

        # LightGBM
        lgbm = lgb.LGBMClassifier(
            n_estimators=400,
            learning_rate=0.04,
            max_depth=6,
            num_leaves=31,
            min_child_samples=20,
            scale_pos_weight=spw,
            random_state=RANDOM_STATE,
            verbose=-1,
        )
        lgbm.fit(X_train, y_train)
        logger.info("LightGBM training complete.")

        # Logistic Regression (with scaling)
        lr = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced",
                C=0.5,
                max_iter=1000,
                random_state=RANDOM_STATE,
            )),
        ])
        lr.fit(X_train, y_train)
        logger.info("Logistic Regression training complete.")

        return {"lgbm": lgbm, "lr_pipeline": lr}

    except Exception as e:
        logger.error(f"Ensemble training failed: {e}", exc_info=True)
        raise


# ---------------------------------------------------------------------------
# Step 3 — Ensemble Predict Proba
# ---------------------------------------------------------------------------
def ensemble_proba(models: dict, X: pd.DataFrame) -> np.ndarray:
    """
    Average predicted probabilities from both ensemble members.

    Parameters
    ----------
    models : dict  Keys: 'lgbm', 'lr_pipeline'
    X : pd.DataFrame

    Returns
    -------
    np.ndarray  Averaged fraud probability per sample.
    """
    if models is None:
        raise ValueError("Cannot predict: ensemble models are None.")

    p_lgbm = models["lgbm"].predict_proba(X)[:, 1]
    p_lr   = models["lr_pipeline"].predict_proba(X)[:, 1]
    return (p_lgbm + p_lr) / 2.0


# ---------------------------------------------------------------------------
# Step 4 — Precision-Focused Threshold Tuning
# ---------------------------------------------------------------------------
def tune_threshold_strict(
    models: dict,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> float:
    """
    Find the lowest threshold that keeps FPR <= TARGET_FPR (3%).

    Uses a fine grid search (0.01 step) over [0.3, 0.99] to find the
    threshold maximising recall while strictly respecting the FPR cap.

    Parameters
    ----------
    models : dict
    X_val : pd.DataFrame
    y_val : pd.Series

    Returns
    -------
    float  Optimal threshold.
    """
    try:
        proba = ensemble_proba(models, X_val)
        best_thresh  = 0.5
        best_recall  = 0.0

        for thresh in np.arange(0.10, 0.99, 0.01):
            preds = (proba >= thresh).astype(int)
            tn = int(((preds == 0) & (y_val == 0)).sum())
            fp = int(((preds == 1) & (y_val == 0)).sum())
            fn = int(((preds == 0) & (y_val == 1)).sum())
            tp = int(((preds == 1) & (y_val == 1)).sum())

            fpr    = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            if fpr <= TARGET_FPR and recall > best_recall:
                best_recall = recall
                best_thresh = thresh

        logger.info(
            f"Tuned threshold: {best_thresh:.2f} | "
            f"Best recall at FPR<={TARGET_FPR:.0%}: {best_recall:.4f}"
        )
        return float(best_thresh)

    except Exception as e:
        logger.error(f"Threshold tuning failed: {e}", exc_info=True)
        raise


# ---------------------------------------------------------------------------
# Step 5 — Evaluate + Compare vs Baseline
# ---------------------------------------------------------------------------
def evaluate_and_compare(
    models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
) -> dict:
    """
    Evaluate Task 13 ensemble at tuned threshold and compare vs Task 11 baseline.

    Parameters
    ----------
    models : dict
    X_test : pd.DataFrame
    y_test : pd.Series
    threshold : float

    Returns
    -------
    dict  Full metrics + comparison deltas.
    """
    try:
        proba = ensemble_proba(models, X_test)

        # Guard: invalid model output
        if np.any(np.isnan(proba)) or np.any(np.isinf(proba)):
            logger.warning("Invalid probability outputs detected. Clipping.")
            proba = np.clip(np.nan_to_num(proba, nan=0.0, posinf=1.0), 0.0, 1.0)

        preds = (proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()

        fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        tpr       = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        roc_auc   = roc_auc_score(y_test, proba)
        avg_prec  = average_precision_score(y_test, proba)

        # Scale baseline FP count to same test size for fair comparison
        total_innocent_test = int(tn + fp)
        baseline_fp_scaled  = round(BASELINE["false_positive_rate"] * total_innocent_test)
        fp_reduction_pct    = (
            (baseline_fp_scaled - fp) / baseline_fp_scaled * 100
            if baseline_fp_scaled > 0 else 0.0
        )

        metrics = {
            "task": "Task 13 — Proctoring FP Reduction (Ship)",
            "timestamp": datetime.now().isoformat(),
            "n_features": len(FEATURE_COLS),
            "model_type": "LightGBM + LR Soft-Voting Ensemble",
            "threshold": round(float(threshold), 4),
            "roc_auc":   round(roc_auc, 4),
            "avg_precision_score": round(avg_prec, 4),
            "precision": round(precision, 4),
            "recall_tpr": round(tpr, 4),
            "false_positive_rate": round(fpr, 4),
            "true_positives":  int(tp),
            "false_positives": int(fp),
            "true_negatives":  int(tn),
            "false_negatives": int(fn),
            "fpr_target_met": bool(fpr <= TARGET_FPR),
            "baseline_comparison": {
                "task11_fpr":            BASELINE["false_positive_rate"],
                "task13_fpr":            round(fpr, 4),
                "fpr_delta":             round(BASELINE["false_positive_rate"] - fpr, 4),
                "task11_precision":      BASELINE["precision"],
                "task13_precision":      round(precision, 4),
                "baseline_fp_scaled":    baseline_fp_scaled,
                "task13_fp":             int(fp),
                "fp_reduction_percent":  round(fp_reduction_pct, 1),
                "fp_reduced":            bool(fp < baseline_fp_scaled),
            },
        }

        logger.info("=" * 55)
        logger.info("  TASK 13 RESULTS vs TASK 11 BASELINE")
        logger.info("=" * 55)
        logger.info(f"  FPR       : {fpr:.2%}  (baseline: {BASELINE['false_positive_rate']:.2%})")
        logger.info(f"  Precision : {precision:.4f}  (baseline: {BASELINE['precision']:.4f})")
        logger.info(f"  Recall    : {tpr:.4f}  (baseline: {BASELINE['recall_tpr']:.4f})")
        logger.info(f"  ROC-AUC   : {roc_auc:.4f}  (baseline: {BASELINE['roc_auc']:.4f})")
        logger.info(f"  FP count  : {fp}  (baseline scaled: ~{baseline_fp_scaled})")
        logger.info(f"  FP Reduction: {fp_reduction_pct:.1f}%")
        logger.info("=" * 55)

        logger.info("\nClassification Report:\n" +
                    classification_report(y_test, preds, target_names=["Innocent", "Fraud"]))

        if metrics["fpr_target_met"]:
            logger.info(f"✅ FPR ({fpr:.2%}) is within strict target ({TARGET_FPR:.0%}).")
        else:
            logger.warning(f"⚠️  FPR ({fpr:.2%}) exceeds target ({TARGET_FPR:.0%}).")

        if metrics["baseline_comparison"]["fp_reduced"]:
            logger.info(f"✅ FALSE POSITIVES REDUCED by {fp_reduction_pct:.1f}% vs Task 11.")
        else:
            logger.warning("⚠️  FP count not reduced vs baseline.")

        with open(METRICS_PATH, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved → {METRICS_PATH}")

        return metrics

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        raise


# ---------------------------------------------------------------------------
# Step 6 — Plot Comparison Chart
# ---------------------------------------------------------------------------
def plot_comparison(metrics: dict) -> None:
    """
    Save a bar chart comparing Task 11 baseline vs Task 13 on key metrics.

    Parameters
    ----------
    metrics : dict  Full metrics dict from evaluate_and_compare().
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        comp = metrics["baseline_comparison"]
        labels_x  = ["FPR (%)", "Precision", "FP Count"]
        baseline_v = [
            comp["task11_fpr"] * 100,
            comp["task11_precision"],
            comp["baseline_fp_scaled"],
        ]
        task13_v = [
            comp["task13_fpr"] * 100,
            comp["task13_precision"],
            comp["task13_fp"],
        ]

        x  = np.arange(len(labels_x))
        w  = 0.35

        fig, ax = plt.subplots(figsize=(9, 5))
        b1 = ax.bar(x - w/2, baseline_v, w, label="Task 11 Baseline", color="#e07b54", alpha=0.88)
        b2 = ax.bar(x + w/2, task13_v,   w, label="Task 13 Ensemble",  color="#3d9e6e", alpha=0.88)

        ax.bar_label(b1, fmt="%.2f", padding=3, fontsize=9)
        ax.bar_label(b2, fmt="%.2f", padding=3, fontsize=9)

        ax.set_title(
            f"Task 13 vs Task 11 — False-Positive Reduction\n"
            f"FP reduced by {comp['fp_reduction_percent']:.1f}%  |  FPR: "
            f"{comp['task11_fpr']:.2%} → {comp['task13_fpr']:.2%}",
            fontsize=12, fontweight="bold"
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels_x, fontsize=11)
        ax.legend(fontsize=10)
        ax.set_ylabel("Value", fontsize=10)
        ax.set_ylim(0, max(baseline_v) * 1.35)
        ax.axhline(3.0, color="red", linestyle="--", linewidth=0.8, label="FPR 3% target")
        plt.tight_layout()
        plt.savefig(CHART_PATH, dpi=150)
        plt.close()
        logger.info(f"Comparison chart saved → {CHART_PATH}")

    except Exception as e:
        logger.warning(f"Chart generation failed (non-fatal): {e}")


# ---------------------------------------------------------------------------
# Step 7 — Save Model
# ---------------------------------------------------------------------------
def save_model(models: dict, threshold: float) -> None:
    """
    Persist the ensemble and threshold to disk.

    Parameters
    ----------
    models : dict
    threshold : float
    """
    try:
        artifact = {
            "ensemble": models,
            "threshold": threshold,
            "features": FEATURE_COLS,
            "task": "Task 13",
        }
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(artifact, f)
        logger.info(f"Model artifact saved → {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Model save failed: {e}", exc_info=True)
        raise


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def run_pipeline() -> None:
    """
    End-to-end Task 13 pipeline:
    1. Generate enhanced proctoring data (7 features)
    2. Train LightGBM + LR soft-voting ensemble
    3. Tune threshold to FPR <= 3%
    4. Evaluate and compare vs Task 11 baseline
    5. Save model, metrics, comparison chart
    """
    logger.info("=" * 60)
    logger.info("  PlaceMux Task 13 — Proctoring FP Reduction (Ship)")
    logger.info("  Baseline (Task 11): FPR=4.74%, FP=9, threshold=0.0")
    logger.info(f"  Target: FPR <= {TARGET_FPR:.0%}, fewer FPs than baseline")
    logger.info("=" * 60)

    df = generate_enhanced_proctoring_data(n_samples=2000, random_state=RANDOM_STATE)

    if df is None or len(df) == 0:
        raise ValueError("Empty dataset — aborting.")

    X = df[FEATURE_COLS]
    y = df["label"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
    )
    logger.info(f"Split — Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    models    = train_ensemble(X_train, y_train)
    threshold = tune_threshold_strict(models, X_val, y_val)
    metrics   = evaluate_and_compare(models, X_test, y_test, threshold)
    plot_comparison(metrics)
    save_model(models, threshold)

    logger.info("Task 13 pipeline complete. ✅")


def main():
    try:
        run_pipeline()
    except FileNotFoundError as e:
        logger.critical(f"Missing required file: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Unhandled fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
