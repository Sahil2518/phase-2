"""
fairness_audit.py — Task 24
Conducts a fairness audit for the PlaceMux Ranker and Proctor models.
Generates synthetic data with protected demographics, computes fairness
metrics (Disparate Impact for Ranker, Equalized Odds / FPR Parity for Proctor),
and signs off the models by updating their metadata if they pass the threshold.
"""

import os
import json
import logging
import datetime
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from data_generator import generate_conversion_data, generate_proctoring_data, FEATURE_COLS

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/task24.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Constants
RANKER_MODEL_PATH = "models/ranker_v2_20260717.pkl"
RANKER_META_PATH = "models/ranker_v2_20260717_metadata.json"
PROCTOR_MODEL_PATH = "models/proctor_v2_task13.pkl"
PROCTOR_META_PATH = "models/proctor_v2_task13_metadata.json"

# Thresholds
FAIRNESS_THRESHOLD = 0.8  # The 80% Rule (Four-Fifths Rule)
PROCTOR_FPR_MAX_DIFF = 0.05  # Max allowable difference in False Positive Rate between groups

def load_model(path: str):
    """
    Safely loads a serialized model from the given path.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model artifact not found at: {path}")
    try:
        model = joblib.load(path)
        logger.info(f"Successfully loaded model from {path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model from {path}: {e}")
        raise

def update_metadata_signoff(meta_path: str, pass_audit: bool):
    """
    Updates the model's metadata file with a fairness sign-off.
    """
    if not os.path.exists(meta_path):
        logger.warning(f"Metadata file not found: {meta_path}. Creating a new one.")
        meta = {}
    else:
        with open(meta_path, "r") as f:
            meta = json.load(f)
            
    meta["fairness_approved"] = pass_audit
    meta["fairness_audit_timestamp"] = datetime.datetime.now().isoformat()
    
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=4)
        
    logger.info(f"Updated metadata sign-off for {meta_path}: Approved={pass_audit}")

def generate_demographic_data(df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    """
    Injects synthetic demographic attributes (Gender, Ethnicity) into a dataframe
    for the purpose of fairness testing.
    """
    rng = np.random.default_rng(random_state)
    df = df.copy()
    
    # 3 groups for Gender, 3 groups for Ethnicity
    df["Gender"] = rng.choice(["Male", "Female", "Non-Binary"], size=len(df), p=[0.48, 0.48, 0.04])
    df["Ethnicity"] = rng.choice(["Group_A", "Group_B", "Group_C"], size=len(df), p=[0.5, 0.3, 0.2])
    
    return df

def audit_ranker(model, df: pd.DataFrame) -> dict:
    """
    Audits the Ranker model for Demographic Parity / Disparate Impact.
    We compute the selection rate (probability of being recommended) for each group.
    
    A candidate is "recommended" if the model's predicted score > 0.5 (for binary interpretation).
    """
    if model is None:
        raise ValueError("Cannot audit: Ranker model is None.")
        
    logger.info("Auditing Ranker model for fairness...")
    try:
        X = df[FEATURE_COLS]
        predictions = model.predict(X)
        # Convert continuous predictions to binary selection (threshold=0.5)
        # Handle potential NaNs/Infs as per standing instructions
        df["predicted_score"] = np.nan_to_num(predictions, nan=0.0, posinf=1.0, neginf=0.0)
        df["selected"] = (df["predicted_score"] >= 0.5).astype(int)
        
        results = {}
        for dem_col in ["Gender", "Ethnicity"]:
            rates = df.groupby(dem_col)["selected"].mean().to_dict()
            max_rate = max(rates.values()) if rates else 0
            
            disparate_impacts = {}
            for group, rate in rates.items():
                di = (rate / max_rate) if max_rate > 0 else 1.0
                disparate_impacts[group] = di
                
            min_di = min(disparate_impacts.values())
            passed = min_di >= FAIRNESS_THRESHOLD
            
            results[dem_col] = {
                "selection_rates": rates,
                "disparate_impact_ratios": disparate_impacts,
                "min_disparate_impact": min_di,
                "passed": passed
            }
            logger.info(f"Ranker - {dem_col} Selection Rates: {rates}")
            logger.info(f"Ranker - {dem_col} Min DI: {min_di:.3f} | Passed: {passed}")
            
        # Plotting
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        sns.barplot(x=list(results["Gender"]["selection_rates"].keys()), y=list(results["Gender"]["selection_rates"].values()))
        plt.title("Ranker Selection Rate by Gender")
        plt.ylim(0, 1)
        
        plt.subplot(1, 2, 2)
        sns.barplot(x=list(results["Ethnicity"]["selection_rates"].keys()), y=list(results["Ethnicity"]["selection_rates"].values()))
        plt.title("Ranker Selection Rate by Ethnicity")
        plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig("logs/fairness_ranker_selection_rates.png")
        plt.close()
        
        return results
    except Exception as e:
        logger.error(f"Ranker audit failed: {e}", exc_info=True)
        raise

def audit_proctor(model, df: pd.DataFrame) -> dict:
    """
    Audits the Proctor model for Equalized Odds (specifically False Positive Rate Parity).
    We ensure that innocent candidates across all demographics have similar rates of being flagged.
    """
    if model is None:
        raise ValueError("Cannot audit: Proctor model is None.")
        
    logger.info("Auditing Proctor model for fairness...")
    try:
        X = df.drop(columns=["label", "Gender", "Ethnicity"]).copy()
        # Ensure all 7 features required by Task 13 proctor model exist
        if "gaze_deviation_score" not in X.columns:
            X["gaze_deviation_score"] = np.random.uniform(0, 1, len(X))
        if "audio_mismatch_score" not in X.columns:
            X["audio_mismatch_score"] = np.random.uniform(0, 1, len(X))
        if "typing_speed_consistency" not in X.columns:
            X["typing_speed_consistency"] = np.random.uniform(0, 1, len(X))
            
        # Ensure column order matches exactly what the model expects
        expected_cols = ['face_match_confidence', 'background_noise_level', 'tab_switch_count', 'keystroke_variance', 'gaze_deviation_score', 'audio_mismatch_score', 'typing_speed_consistency']
        X = X[expected_cols]

        if isinstance(model, dict) and "ensemble" in model:
            # Proctor v2 format (Task 13 soft voting)
            ensemble = model["ensemble"]
            threshold = model.get("threshold", 0.5)
            
            p_lgbm = ensemble["lgbm"].predict_proba(X)[:, 1]
            p_lr = ensemble["lr_pipeline"].predict_proba(X)[:, 1]
            preds = (p_lgbm + p_lr) / 2.0
            
            df["predicted_fraud"] = (np.nan_to_num(preds, nan=0.0) >= threshold).astype(int)
        else:
            # Fallback
            estimator = model
            threshold = 0.5
            if hasattr(estimator, "predict_proba"):
                preds = estimator.predict_proba(X)[:, 1]
                df["predicted_fraud"] = (np.nan_to_num(preds, nan=0.0) >= threshold).astype(int)
            else:
                preds = estimator.predict(X)
                df["predicted_fraud"] = np.nan_to_num(preds, nan=0.0).astype(int)
            
        # Focus on innocent candidates (label == 0) to compute FPR
        innocents = df[df["label"] == 0]
        
        results = {}
        for dem_col in ["Gender", "Ethnicity"]:
            fpr_rates = innocents.groupby(dem_col)["predicted_fraud"].mean().to_dict()
            max_fpr = max(fpr_rates.values()) if fpr_rates else 0
            min_fpr = min(fpr_rates.values()) if fpr_rates else 0
            fpr_diff = max_fpr - min_fpr
            
            passed = fpr_diff <= PROCTOR_FPR_MAX_DIFF
            
            results[dem_col] = {
                "false_positive_rates": fpr_rates,
                "max_fpr_difference": fpr_diff,
                "passed": passed
            }
            logger.info(f"Proctor - {dem_col} Max FPR Diff: {fpr_diff:.3f} | Passed: {passed}")
            
        # Plotting
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        sns.barplot(x=list(results["Gender"]["false_positive_rates"].keys()), y=list(results["Gender"]["false_positive_rates"].values()))
        plt.title("Proctor FPR by Gender")
        plt.ylim(0, 0.2)
        
        plt.subplot(1, 2, 2)
        sns.barplot(x=list(results["Ethnicity"]["false_positive_rates"].keys()), y=list(results["Ethnicity"]["false_positive_rates"].values()))
        plt.title("Proctor FPR by Ethnicity")
        plt.ylim(0, 0.2)
        
        plt.tight_layout()
        plt.savefig("logs/fairness_proctor_fpr_rates.png")
        plt.close()
        
        return results
    except Exception as e:
        logger.error(f"Proctor audit failed: {e}", exc_info=True)
        raise

def run_fairness_audit():
    """
    Main function to execute the fairness audit pipeline.
    """
    logger.info("Starting Fairness Audit...")
    
    report = {
        "timestamp": datetime.datetime.now().isoformat(),
        "ranker": {},
        "proctor": {},
        "overall_signoff": False
    }
    
    try:
        # Load Models
        ranker_model = load_model(RANKER_MODEL_PATH)
        proctor_model = load_model(PROCTOR_MODEL_PATH)
        
        # Generate demographic-aware datasets
        logger.info("Generating synthetic data with demographics...")
        ranker_data_raw = generate_conversion_data(n_samples=10000, random_state=42)
        ranker_data = generate_demographic_data(ranker_data_raw, random_state=88)
        
        proctor_data_raw = generate_proctoring_data(n_samples=10000, random_state=123)
        proctor_data = generate_demographic_data(proctor_data_raw, random_state=88)
        
        # Audit
        ranker_results = audit_ranker(ranker_model, ranker_data)
        proctor_results = audit_proctor(proctor_model, proctor_data)
        
        report["ranker"] = ranker_results
        report["proctor"] = proctor_results
        
        # Evaluate Overall Success
        ranker_passed = all(res["passed"] for res in ranker_results.values())
        proctor_passed = all(res["passed"] for res in proctor_results.values())
        
        # Sign-off
        update_metadata_signoff(RANKER_META_PATH, ranker_passed)
        update_metadata_signoff(PROCTOR_META_PATH, proctor_passed)
        
        report["overall_signoff"] = bool(ranker_passed and proctor_passed)
        
        # Save Report
        with open("logs/fairness_audit_report.json", "w") as f:
            json.dump(report, f, indent=4)
            
        logger.info(f"Fairness Audit complete. Overall Sign-off: {report['overall_signoff']}")
        
    except Exception as e:
        logger.error(f"Fairness Audit aborted due to an error: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    run_fairness_audit()
