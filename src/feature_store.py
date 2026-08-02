"""
feature_store.py — Feature Store for PlaceMux MLOps (Task 23)

Provides a lightweight SQLite-based feature store to ingest and serve
precomputed features for students and jobs.

Standing instructions: robust error handling, structured logging.
"""

import os
import json
import sqlite3
import logging
import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

DB_PATH = "data/mlops.db"

def _get_connection() -> sqlite3.Connection:
    """Helper to establish SQLite connection."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH)

def init_feature_store():
    """
    Initialize the feature store database schema.
    """
    try:
        with _get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS features (
                    entity_type TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    features_json TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (entity_type, entity_id)
                )
            """)
            conn.commit()
        logger.info("Feature store initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize feature store: {e}", exc_info=True)
        raise

def put_features(entity_type: str, entity_id: str, features: Dict[str, Any]):
    """
    Ingest features for a specific entity into the store.
    
    Parameters
    ----------
    entity_type : str
        The type of entity (e.g., "student" or "job").
    entity_id : str
        The unique identifier for the entity.
    features : dict
        A dictionary containing the feature keys and values.
    """
    try:
        if not entity_id:
            raise ValueError(f"Invalid entity_id provided for {entity_type}")
            
        features_json = json.dumps(features)
        
        with _get_connection() as conn:
            cursor = conn.cursor()
            # Upsert logic (insert or replace)
            cursor.execute("""
                INSERT INTO features (entity_type, entity_id, features_json, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(entity_type, entity_id) DO UPDATE SET
                    features_json=excluded.features_json,
                    updated_at=CURRENT_TIMESTAMP
            """, (entity_type, entity_id, features_json))
            conn.commit()
            
        logger.debug(f"Ingested features for {entity_type} {entity_id}")
    except Exception as e:
        logger.error(f"Failed to put features for {entity_type} {entity_id}: {e}", exc_info=True)
        raise

def get_features(entity_type: str, entity_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve features for a specific entity from the store.
    
    Parameters
    ----------
    entity_type : str
        The type of entity (e.g., "student" or "job").
    entity_id : str
        The unique identifier for the entity.
        
    Returns
    -------
    dict
        The feature dictionary, or None if not found.
    """
    try:
        with _get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT features_json FROM features 
                WHERE entity_type = ? AND entity_id = ?
            """, (entity_type, entity_id))
            
            row = cursor.fetchone()
            if row:
                return json.loads(row[0])
            else:
                logger.warning(f"No features found for {entity_type} {entity_id}")
                return None
    except Exception as e:
        logger.error(f"Failed to get features for {entity_type} {entity_id}: {e}", exc_info=True)
        raise

# Helper wrappers for clarity
def put_student_features(student_id: str, features: Dict[str, Any]):
    put_features("student", student_id, features)

def get_student_features(student_id: str) -> Optional[Dict[str, Any]]:
    return get_features("student", student_id)

def put_job_features(job_id: str, features: Dict[str, Any]):
    put_features("job", job_id, features)

def get_job_features(job_id: str) -> Optional[Dict[str, Any]]:
    return get_features("job", job_id)
