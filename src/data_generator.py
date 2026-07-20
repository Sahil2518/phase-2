"""
data_generator.py — Synthetic Training Data Generator for Task 03 Ranker

Generates synthetic (student, job) feature pairs labelled with a reference
relevance score. Used to cold-start the ML ranking model when real human
feedback data is unavailable.

Standing instructions: robust error handling, structured logging.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Tuple

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/task03.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature column names (must match ranker.py FEATURE_COLS)
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "skill_overlap_ratio",
    "norm_experience_gap",
    "capped_salary_ratio",
    "education_met",
    "coding_threshold_met",
    "communication_threshold_met",
    "location_match",
]

# Reference weights — used to compute synthetic ground-truth labels
WEIGHTS = {
    "skill_overlap_ratio":          0.40,
    "norm_experience_gap":          0.20,
    "capped_salary_ratio":          0.15,
    "education_met":                0.10,
    "coding_threshold_met":         0.10,
    "communication_threshold_met":  0.05,
    "location_match":               0.02,   # bonus; weights sum to 1.02; normalised below
}
WEIGHT_SUM = sum(WEIGHTS.values())  # 1.02

# Task 7: Conversion-optimised weights
CONVERSION_WEIGHTS = {
    "skill_overlap_ratio":          0.45,   # Increased: Top conversion blocker
    "norm_experience_gap":          0.20,   # Unchanged
    "capped_salary_ratio":          0.20,   # Increased: Students pay for salary-aligned roles
    "education_met":                0.07,   # Decreased: Less binary blocking
    "coding_threshold_met":         0.05,   # Decreased: Many paid apps bypass this
    "communication_threshold_met":  0.02,   # Decreased: Minor signal
    "location_match":               0.01,   # Decreased: Remote-work era reduces weight
}
CONVERSION_WEIGHT_SUM = sum(CONVERSION_WEIGHTS.values())

def _reference_label(row: pd.Series) -> float:
    """Compute a ground-truth relevance score from feature values."""
    score = sum(WEIGHTS[col] * row[col] for col in FEATURE_COLS)
    return float(np.clip(score / WEIGHT_SUM, 0.0, 1.0))

def _conversion_label(row: pd.Series) -> float:
    """Compute a conversion-optimised ground-truth score from feature values."""
    score = sum(CONVERSION_WEIGHTS[col] * row[col] for col in FEATURE_COLS)
    return float(np.clip(score / CONVERSION_WEIGHT_SUM, 0.0, 1.0))


def generate_synthetic_data(
    n_samples: int = 600,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate n_samples random (student, job) feature vectors and their labels.

    Parameters
    ----------
    n_samples : int
        Number of synthetic records to create. Default 600.
    random_state : int
        NumPy random seed for reproducibility. Default 42.

    Returns
    -------
    pd.DataFrame with columns = FEATURE_COLS + ['label']
    """
    try:
        rng = np.random.default_rng(random_state)
        logger.info(f"Generating {n_samples} synthetic training records (seed={random_state})...")

        # --- continuous features ---
        skill_overlap   = rng.uniform(0.0, 1.0, n_samples)
        # experience gap: student_exp - job_min_exp, range ~ [-2, +6]
        # normalise to [0,1] via (gap+2)/4, then clip
        raw_exp_gap     = rng.uniform(-2.0, 6.0, n_samples)
        norm_exp_gap    = np.clip((raw_exp_gap + 2.0) / 4.0, 0.0, 1.0)

        # salary_max / expected_salary, range ~ [0.3, 2.0], cap at 1.0
        raw_salary      = rng.uniform(0.3, 2.0, n_samples)
        capped_salary   = np.clip(raw_salary, 0.0, 1.0)

        # --- binary features ---
        edu_met         = rng.integers(0, 2, n_samples).astype(float)
        coding_met      = rng.integers(0, 2, n_samples).astype(float)
        comm_met        = rng.integers(0, 2, n_samples).astype(float)
        loc_match       = rng.integers(0, 2, n_samples).astype(float)

        df = pd.DataFrame({
            "skill_overlap_ratio":         skill_overlap,
            "norm_experience_gap":         norm_exp_gap,
            "capped_salary_ratio":         capped_salary,
            "education_met":               edu_met,
            "coding_threshold_met":        coding_met,
            "communication_threshold_met": comm_met,
            "location_match":              loc_match,
        })

        df["label"] = df.apply(_reference_label, axis=1)

        assert len(df) == n_samples, "Row count mismatch after generation."
        assert df.isnull().sum().sum() == 0, "NaN values found in synthetic data."
        assert df["label"].between(0.0, 1.0).all(), "Label out of [0,1] range."

        logger.info(
            f"Synthetic data generated. "
            f"Label stats — mean: {df['label'].mean():.4f}, "
            f"std: {df['label'].std():.4f}, "
            f"min: {df['label'].min():.4f}, "
            f"max: {df['label'].max():.4f}"
        )
        return df

    except Exception as e:
        logger.error(f"Synthetic data generation failed: {e}", exc_info=True)
        raise

def generate_conversion_data(
    n_samples: int = 1200,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic data specifically labelled for conversion optimisation.
    This calls generate_synthetic_data() but replaces the 'label' column
    using _conversion_label().
    
    Parameters
    ----------
    n_samples : int
        Number of synthetic records to create.
    random_state : int
        NumPy random seed for reproducibility.

    Returns
    -------
    pd.DataFrame with conversion-optimised labels.
    """
    try:
        logger.info("Generating conversion-optimised dataset...")
        df = generate_synthetic_data(n_samples=n_samples, random_state=random_state)
        df["label"] = df.apply(_conversion_label, axis=1)
        
        assert len(df) == n_samples, "Row count mismatch."
        assert df["label"].between(0.0, 1.0).all(), "Conversion label out of [0,1] range."
        
        logger.info(
            f"Conversion labels applied. "
            f"Stats — mean: {df['label'].mean():.4f}, "
            f"std: {df['label'].std():.4f}, "
            f"min: {df['label'].min():.4f}, "
            f"max: {df['label'].max():.4f}"
        )
        return df
    except Exception as e:
        logger.error(f"Conversion data generation failed: {e}", exc_info=True)
        raise


def match_vector_to_feature_row(mv) -> dict:
    """
    Convert a MatchVector (from match_vectors.py) to the flat feature dict
    expected by the ranker model.

    Parameters
    ----------
    mv : MatchVector (dict or Pydantic model with .model_dump())
    """
    try:
        d = mv if isinstance(mv, dict) else mv.model_dump()

        raw_exp_gap    = d["experience_gap"]
        norm_exp_gap   = float(np.clip((raw_exp_gap + 2.0) / 4.0, 0.0, 1.0))

        raw_salary     = d["salary_match_ratio"]
        capped_salary  = float(np.clip(raw_salary, 0.0, 1.0))

        return {
            "skill_overlap_ratio":          float(d["skill_overlap_ratio"]),
            "norm_experience_gap":          norm_exp_gap,
            "capped_salary_ratio":          capped_salary,
            "education_met":                float(d["education_met"]),
            "coding_threshold_met":         float(d["coding_threshold_met"]),
            "communication_threshold_met":  float(d["communication_threshold_met"]),
            "location_match":               float(d["location_match"]),
        }
    except KeyError as e:
        logger.error(f"Missing key when converting MatchVector to feature row: {e}")
        raise


if __name__ == "__main__":
    df = generate_synthetic_data()
    print(df.describe())

# ---------------------------------------------------------------------------
# Task 11 — Proctoring Data Generator
# ---------------------------------------------------------------------------
def generate_proctoring_data(
    n_samples: int = 1500,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic proctoring events with a 'fraud' label.
    Simulates signals like face_match_confidence, background_noise, and tab_switches.
    """
    try:
        rng = np.random.default_rng(random_state)
        logger.info(f"Generating {n_samples} synthetic proctoring records...")

        # Most candidates are innocent (~85%), some commit fraud (~15%)
        labels = rng.binomial(1, 0.15, n_samples)
        
        # Innocent distributions
        # High face match, low noise, few tab switches, low keystroke variance
        face_match_innocent = rng.normal(0.95, 0.05, n_samples)
        noise_innocent = rng.exponential(0.1, n_samples)
        tabs_innocent = rng.poisson(1, n_samples)
        keystroke_innocent = rng.normal(0.2, 0.1, n_samples)
        
        # Fraud distributions
        # Lower/erratic face match, higher noise, many tab switches, erratic keystrokes
        face_match_fraud = rng.normal(0.60, 0.2, n_samples)
        noise_fraud = rng.exponential(0.4, n_samples)
        tabs_fraud = rng.poisson(8, n_samples)
        keystroke_fraud = rng.normal(0.7, 0.2, n_samples)
        
        face_match = np.where(labels == 1, face_match_fraud, face_match_innocent)
        noise = np.where(labels == 1, noise_fraud, noise_innocent)
        tabs = np.where(labels == 1, tabs_fraud, tabs_innocent)
        keystroke = np.where(labels == 1, keystroke_fraud, keystroke_innocent)

        df = pd.DataFrame({
            "face_match_confidence": np.clip(face_match, 0.0, 1.0),
            "background_noise_level": np.clip(noise, 0.0, 1.0),
            "tab_switch_count": np.clip(tabs, 0, 50).astype(int),
            "keystroke_variance": np.clip(keystroke, 0.0, 1.0),
            "label": labels
        })
        
        logger.info(f"Proctoring data generated. Fraud rate: {df['label'].mean():.2%}")
        return df

    except Exception as e:
        logger.error(f"Proctoring data generation failed: {e}", exc_info=True)
        raise
