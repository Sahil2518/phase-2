"""
model_registry.py — Model Registry for PlaceMux MLOps (Task 23)

Provides a lightweight SQLite-based model registry to track model versions,
metadata, and deployment stages (e.g., "Staging", "Production").

Standing instructions: robust error handling, structured logging.
"""

import os
import json
import sqlite3
import logging
import datetime
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

DB_PATH = "data/mlops.db"

def _get_connection() -> sqlite3.Connection:
    """Helper to establish SQLite connection."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH)

def init_registry():
    """
    Initialize the model registry database schema.
    """
    try:
        with _get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    artifact_path TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    features TEXT NOT NULL,
                    metrics TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(model_name, version)
                )
            """)
            conn.commit()
        logger.info("Model registry initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize model registry: {e}", exc_info=True)
        raise

def register_model(model_name: str, artifact_path: str, metrics: Dict[str, Any], features: list) -> int:
    """
    Register a new version of a model.
    
    Parameters
    ----------
    model_name : str
        The base name of the model (e.g., "ranker").
    artifact_path : str
        Path to the serialized model file (.pkl).
    metrics : dict
        Dictionary of evaluation metrics.
    features : list
        List of feature column names the model expects.
        
    Returns
    -------
    version : int
        The newly assigned version number.
    """
    try:
        if not os.path.exists(artifact_path):
            raise FileNotFoundError(f"Artifact not found at {artifact_path}")

        features_json = json.dumps(features)
        metrics_json = json.dumps(metrics)
        stage = "None"

        with _get_connection() as conn:
            cursor = conn.cursor()
            
            # Determine next version
            cursor.execute("SELECT MAX(version) FROM models WHERE model_name = ?", (model_name,))
            row = cursor.fetchone()
            next_version = 1 if row[0] is None else row[0] + 1
            
            cursor.execute("""
                INSERT INTO models (model_name, version, artifact_path, stage, features, metrics)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (model_name, next_version, artifact_path, stage, features_json, metrics_json))
            conn.commit()
            
        logger.info(f"Registered {model_name} version {next_version} at {artifact_path}")
        return next_version
    except Exception as e:
        logger.error(f"Failed to register model {model_name}: {e}", exc_info=True)
        raise

def transition_model_stage(model_name: str, version: int, stage: str):
    """
    Promote or demote a model version to a specific deployment stage.
    Ensures only one version of a model occupies the 'Production' stage.
    
    Parameters
    ----------
    model_name : str
        The base name of the model.
    version : int
        The version to transition.
    stage : str
        The target stage (e.g., "Production", "Staging", "Archived").
    """
    try:
        with _get_connection() as conn:
            cursor = conn.cursor()
            
            # Check if version exists
            cursor.execute("SELECT id FROM models WHERE model_name = ? AND version = ?", (model_name, version))
            if cursor.fetchone() is None:
                raise ValueError(f"Model {model_name} version {version} does not exist.")
            
            if stage == "Production":
                # Demote current production model if any
                cursor.execute("""
                    UPDATE models SET stage = 'Archived' 
                    WHERE model_name = ? AND stage = 'Production'
                """, (model_name,))
                
            # Promote target model
            cursor.execute("""
                UPDATE models SET stage = ? 
                WHERE model_name = ? AND version = ?
            """, (stage, model_name, version))
            
            conn.commit()
            
        logger.info(f"Transitioned {model_name} version {version} to stage '{stage}'")
    except Exception as e:
        logger.error(f"Failed to transition model stage: {e}", exc_info=True)
        raise

def get_model(model_name: str, stage: str = "Production") -> Tuple[str, Dict[str, Any], list]:
    """
    Retrieve the artifact path and metadata for a model in a specific stage.
    
    Parameters
    ----------
    model_name : str
        The base name of the model.
    stage : str
        The stage to retrieve (default "Production").
        
    Returns
    -------
    artifact_path : str
    metrics : dict
    features : list
    """
    try:
        with _get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT artifact_path, metrics, features, version 
                FROM models 
                WHERE model_name = ? AND stage = ?
                ORDER BY version DESC LIMIT 1
            """, (model_name, stage))
            row = cursor.fetchone()
            
            if row is None:
                raise ValueError(f"No model found for {model_name} in stage '{stage}'")
                
            artifact_path, metrics_json, features_json, version = row
            logger.info(f"Retrieved {model_name} version {version} (Stage: {stage})")
            
            return artifact_path, json.loads(metrics_json), json.loads(features_json)
    except Exception as e:
        logger.error(f"Failed to retrieve model {model_name} (stage {stage}): {e}", exc_info=True)
        raise
