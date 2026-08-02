"""
train_task23.py — End-to-End MLOps Foundation Demo (Task 23)

This script demonstrates the PlaceMux MLOps foundations:
1. Initializes the Model Registry and Feature Store.
2. Trains a ranking model and saves its artifact.
3. Registers the model and promotes it to 'Production'.
4. Ingests synthetic features into the Feature Store.
5. Simulates an inference request by querying the Feature Store
   and loading the Production model from the Registry.

Standing instructions: robust error handling, structured logging.
"""

import os
import sys
import logging
import pandas as pd
from typing import Any

from src.data_generator import generate_synthetic_data, FEATURE_COLS
from src.ranker import train_ranker, save_ranker, load_ranker
from src.model_registry import init_registry, register_model, transition_model_stage, get_model
from src.feature_store import init_feature_store, put_student_features, put_job_features, get_student_features, get_job_features

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/task23_mlops.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

def run_mlops_pipeline():
    """Execute the end-to-end MLOps demonstration."""
    logger.info("Starting MLOps Foundation Pipeline (Task 23)")

    # 1. Initialize MLOps Infrastructure
    init_registry()
    init_feature_store()

    # 2. Train Model
    logger.info("Step 1: Generating training data & training model...")
    df_train = generate_synthetic_data(n_samples=500, random_state=42)
    model, metrics = train_ranker(df_train)

    # 3. Save Model Artifact
    logger.info("Step 2: Serializing model artifact...")
    artifact_path = save_ranker(model, metrics)

    # 4. Register Model and Promote to Production
    logger.info("Step 3: Registering model in Model Registry...")
    model_name = "ranker_v1"
    version = register_model(
        model_name=model_name,
        artifact_path=artifact_path,
        metrics=metrics,
        features=FEATURE_COLS
    )
    
    logger.info("Promoting model to 'Production' stage...")
    transition_model_stage(model_name, version, "Production")

    # 5. Ingest Features into Feature Store
    logger.info("Step 4: Ingesting features into Feature Store...")
    # Grab the first row of our synthetic data to use as a test case
    sample_features = df_train.iloc[0][FEATURE_COLS].to_dict()
    student_id = "STU-9999"
    job_id = "JOB-1111"
    
    # We simulate that the student provides some features and the job provides others.
    # For this demo, we just store partial sets and combine them, or just store the 
    # flat feature row directly as a "precomputed pair feature" for simplicity.
    # In a real system, you'd store student profile data and job profile data, 
    # then compute the match vector online.
    put_student_features(student_id, {"profile_completeness": 0.95, "years_exp": 3})
    put_job_features(job_id, {"min_exp": 2, "remote_allowed": True})
    
    # Let's also just store the precomputed match features to test the ranker directly
    # Using a composite key for the precomputed pair features
    from src.feature_store import put_features, get_features
    put_features("match_vector", f"{student_id}_{job_id}", sample_features)

    # 6. Simulate Online Inference
    logger.info("Step 5: Simulating Online Inference via MLOps Infrastructure...")
    
    # Fetch Production Model
    prod_artifact, prod_metrics, prod_features = get_model(model_name, stage="Production")
    logger.info(f"Loaded Production Model Artifact: {prod_artifact}")
    prod_model = load_ranker(prod_artifact)
    
    # Fetch Features
    retrieved_features = get_features("match_vector", f"{student_id}_{job_id}")
    if retrieved_features is None:
        raise ValueError("Failed to retrieve features from Feature Store.")
        
    logger.info(f"Retrieved Features: {retrieved_features}")
    
    # Predict
    # Ranker expects a DataFrame matching FEATURE_COLS
    X_test = pd.DataFrame([retrieved_features], columns=prod_features)
    import numpy as np
    raw_score = float(prod_model.predict(X_test)[0])
    
    # Edge case guard from standing instructions
    if np.isnan(raw_score) or np.isinf(raw_score):
        logger.warning(f"Invalid model output ({raw_score}). Defaulting to 0.0.")
        final_score = 0.0
    else:
        final_score = float(np.clip(raw_score, 0.0, 1.0))
        
    logger.info(f"Inference complete. Predicted Match Score: {final_score:.4f}")
    logger.info("MLOps Foundation Pipeline completed successfully.")

def main():
    """Main entry point with top-level error trap."""
    try:
        run_mlops_pipeline()
    except Exception as e:
        logger.critical(f"Unhandled fatal error in pipeline: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
