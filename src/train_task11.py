"""
train_task11.py - Task 11: Proctoring Hardening / False-Positive Reduction

Focus: Train a LightGBM classifier on synthetic proctoring events and tune
the decision threshold to minimize false positives (innocent candidates being
wrongly flagged as fraudsters), keeping FPR strictly below 5%.

Standing instructions: robust error handling, structured logging, random_state=42.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
)

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/task11.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "face_match_confidence",
    "background_noise_level",
    "tab_switch_count",
    "keystroke_variance",
]
TARGET_FPR_THRESHOLD = 0.05  # Must keep False Positive Rate below 5%
RANDOM_STATE = 42
MODEL_PATH = "models/proctor_classifier_v1.pkl"
METRICS_PATH = "logs/task11_metrics.json"


# ---------------------------------------------------------------------------
# Step 1: Load Data
# ---------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    """
    Load synthetic proctoring data from data_generator.py.

    Returns
    -------
    pd.DataFrame
        DataFrame with features and binary 'label' column.
    """
    try:
        logger.info("Loading synthetic proctoring data...")
        # Import here to avoid circular dependency issues
        sys.path.insert(0, os.path.dirname(__file__))
        from data_generator import generate_proctoring_data

        df = generate_proctoring_data(n_samples=1500, random_state=RANDOM_STATE)

        assert len(df) > 0, "Empty dataset returned from generator."
        assert "label" in df.columns, "Missing 'label' column in dataset."
        assert not df.isnull().any().any(), "NaN values found in dataset."

        logger.info(
            f"Data loaded. Shape: {df.shape} | "
            f"Fraud rate: {df['label'].mean():.2%}"
        )
        return df
    except Exception as e:
        logger.critical(f"Failed to load data: {e}", exc_info=True)
        raise


# ---------------------------------------------------------------------------
# Step 2: Train Model
# ---------------------------------------------------------------------------
def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> lgb.LGBMClassifier:
    """
    Train a LightGBM binary classifier on proctoring features.

    Uses scale_pos_weight to handle class imbalance (~85% innocent, ~15% fraud).

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Binary labels (1 = fraud, 0 = innocent).

    Returns
    -------
    lgb.LGBMClassifier
        Trained classifier.
    """
    try:
        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()
        scale_pos_weight = neg / pos if pos > 0 else 1.0
        logger.info(
            f"Class distribution — Innocent: {neg}, Fraud: {pos} | "
            f"scale_pos_weight: {scale_pos_weight:.2f}"
        )

        model = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            verbose=-1,
        )
        model.fit(X_train, y_train)
        logger.info("Model training complete.")
        return model
    except Exception as e:
        logger.error(f"Model training failed: {e}", exc_info=True)
        raise


# ---------------------------------------------------------------------------
# Step 3: Tune Threshold for False-Positive Reduction
# ---------------------------------------------------------------------------
def tune_threshold(
    model: lgb.LGBMClassifier,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> float:
    """
    Find the decision threshold that keeps False Positive Rate (FPR) below
    TARGET_FPR_THRESHOLD while maximizing Recall on true fraud cases.

    A false positive means an innocent candidate is flagged as a fraudster —
    which is the primary harm we want to minimize in this trust layer.

    Parameters
    ----------
    model : lgb.LGBMClassifier
        Trained classifier.
    X_val : pd.DataFrame
        Validation feature matrix.
    y_val : pd.Series
        True binary labels.

    Returns
    -------
    float
        The optimal threshold value.
    """
    try:
        proba = model.predict_proba(X_val)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y_val, proba)

        best_threshold = 0.5  # fallback
        best_recall = 0.0

        for thresh in thresholds:
            preds = (proba >= thresh).astype(int)
            tn = ((preds == 0) & (y_val == 0)).sum()
            fp = ((preds == 1) & (y_val == 0)).sum()
            fn = ((preds == 0) & (y_val == 1)).sum()
            tp = ((preds == 1) & (y_val == 1)).sum()

            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            # Only accept threshold if FPR is below our target
            if fpr <= TARGET_FPR_THRESHOLD and recall > best_recall:
                best_recall = recall
                best_threshold = thresh

        logger.info(
            f"Optimal threshold found: {best_threshold:.4f} | "
            f"Best Recall at FPR<={TARGET_FPR_THRESHOLD:.0%}: {best_recall:.4f}"
        )
        return float(best_threshold)
    except Exception as e:
        logger.error(f"Threshold tuning failed: {e}", exc_info=True)
        raise


# ---------------------------------------------------------------------------
# Step 4: Evaluate and Save Metrics
# ---------------------------------------------------------------------------
def evaluate(
    model: lgb.LGBMClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
) -> dict:
    """
    Evaluate the model at the tuned threshold and log all key metrics.

    Parameters
    ----------
    model : lgb.LGBMClassifier
    X_test : pd.DataFrame
    y_test : pd.Series
    threshold : float
        The optimal decision threshold.

    Returns
    -------
    dict
        Dictionary of all key metrics.
    """
    try:
        proba = model.predict_proba(X_test)[:, 1]
        preds = (proba >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        roc_auc = roc_auc_score(y_test, proba)

        metrics = {
            "task": "Task 11 — Proctoring Hardening",
            "timestamp": datetime.now().isoformat(),
            "threshold": round(threshold, 4),
            "roc_auc": round(roc_auc, 4),
            "precision": round(precision, 4),
            "recall_tpr": round(tpr, 4),
            "false_positive_rate": round(fpr, 4),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
            "fpr_target_met": bool(fpr <= TARGET_FPR_THRESHOLD),
        }

        logger.info("--- Evaluation Results ---")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v}")

        logger.info("\nClassification Report:")
        logger.info(
            "\n" + classification_report(y_test, preds, target_names=["Innocent", "Fraud"])
        )

        if not metrics["fpr_target_met"]:
            logger.warning(
                f"FPR ({fpr:.2%}) exceeds target ({TARGET_FPR_THRESHOLD:.0%})! "
                f"Further threshold adjustment may be needed."
            )
        else:
            logger.info(
                f"FPR ({fpr:.2%}) is within target. False-positive reduction: PASSED."
            )

        with open(METRICS_PATH, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to {METRICS_PATH}")

        return metrics
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        raise


# ---------------------------------------------------------------------------
# Step 5: Save Model
# ---------------------------------------------------------------------------
def save_model(model: lgb.LGBMClassifier, threshold: float) -> None:
    """
    Serialize the trained model and threshold to disk.

    Parameters
    ----------
    model : lgb.LGBMClassifier
    threshold : float
    """
    try:
        import pickle

        artifact = {"model": model, "threshold": threshold}
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(artifact, f)
        logger.info(f"Model artifact saved to {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Model save failed: {e}", exc_info=True)
        raise


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def run_pipeline() -> None:
    """
    End-to-end proctoring hardening pipeline:
    1. Load data
    2. Train LightGBM classifier
    3. Tune threshold to minimize false positives (FPR < 5%)
    4. Evaluate on test set
    5. Save model artifact and metrics
    """
    logger.info("=" * 60)
    logger.info("  PlaceMux Task 11 — Proctoring Hardening")
    logger.info("  Focus: False-Positive Reduction")
    logger.info("=" * 60)

    df = load_data()

    X = df[FEATURE_COLS]
    y = df["label"]

    # 70% train, 15% val (threshold tuning), 15% test (final eval)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
    )

    logger.info(
        f"Split — Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}"
    )

    model = train_model(X_train, y_train)
    threshold = tune_threshold(model, X_val, y_val)
    evaluate(model, X_test, y_test, threshold)
    save_model(model, threshold)

    logger.info("Task 11 pipeline complete.")


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
