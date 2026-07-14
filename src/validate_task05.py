"""
validate_task05.py — Task 05: Matching Validation Suite

This module provides an automated integration test suite for the Company Portal API.
It spins up the FastAPI application using TestClient, loads the real-world dataset,
and validates that the ranking and explainability payloads flow through the HTTP
boundary correctly.

Test Cases:
1. Endpoint health check (200 OK).
2. Rank candidates for a given job (Validates sorting, shortlisting logic, and schemas).
3. Edge case: Missing or empty candidate lists.
"""

import os
import sys
import json
import logging
import pytest
from fastapi.testclient import TestClient

# Import the API app
from src.api import app
from src.model_schemas import StudentFeatures, JobFeatures

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/task05_validation.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("PlaceMux-Validation")

client = TestClient(app)

# ---------------------------------------------------------------------------
# Data Loading Fixtures
# ---------------------------------------------------------------------------
def load_data():
    """Helper to load real-world JSON datasets for validation."""
    if not os.path.exists("data/sample_jobs.json") or not os.path.exists("data/sample_students.json"):
        logger.error("Test data files missing. Please run fetch_real_world_data.py first.")
        sys.exit(1)
        
    with open("data/sample_jobs.json", "r") as f:
        jobs = json.load(f)
    with open("data/sample_students.json", "r") as f:
        students = json.load(f)
        
    return jobs, students

# ---------------------------------------------------------------------------
# Validation Tests
# ---------------------------------------------------------------------------

def test_health_check():
    """Verify that the API and the underlying ML model are loaded and healthy."""
    logger.info("Executing test_health_check...")
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200, "Health check failed"
        data = response.json()
        assert data["status"] == "online"
        # Ensure the ML model actually loaded successfully in memory
        assert data["model_loaded"] is True, "Ranker model failed to load in API"
    logger.info("Health check passed.")


def test_rank_candidates_end_to_end():
    """
    Core Validation: Sends a real job and real candidates to the API.
    Asserts HTTP 200, checks sorting, and validates the presence of the 
    explainability payload.
    """
    logger.info("Executing test_rank_candidates_end_to_end...")
    jobs, students = load_data()
    
    # Pick the first job and all students
    target_job = jobs[0]
    
    payload = {
        "job": target_job,
        "candidates": students
    }
    
    with TestClient(app) as client:
        response = client.post("/api/v1/portal/rank-candidates", json=payload)
    
    assert response.status_code == 200, f"API error: {response.text}"
    
    data = response.json()
    
    # Validate Schema structural integrity
    assert data["job_id"] == target_job["job_id"]
    assert data["total_evaluated"] == len(students)
    
    ranked = data["ranked_candidates"]
    assert len(ranked) == len(students)
    
    # Validate Sorting (Must be descending by score)
    scores = [c["score"] for c in ranked]
    assert scores == sorted(scores, reverse=True), "Candidates are not sorted by score!"
    
    # Validate Explainability Payload
    top_candidate = ranked[0]
    assert "explanation_payload" in top_candidate
    payload_data = top_candidate["explanation_payload"]
    assert "summary" in payload_data
    assert "confidence" in payload_data
    assert "shortlist" in payload_data
    
    logger.info("test_rank_candidates_end_to_end passed.")


def test_rank_jobs_end_to_end():
    """
    Core Validation: Sends a real student and real jobs to the API.
    Asserts HTTP 200 and structural correctness.
    """
    logger.info("Executing test_rank_jobs_end_to_end...")
    jobs, students = load_data()
    
    target_student = students[0]
    
    payload = {
        "student": target_student,
        "jobs": jobs
    }
    
    with TestClient(app) as client:
        response = client.post("/api/v1/marketplace/rank-jobs", json=payload)
        
    assert response.status_code == 200, f"API error: {response.text}"
    
    data = response.json()
    assert data["student_id"] == target_student["student_id"]
    
    ranked = data["ranked_jobs"]
    scores = [j["score"] for j in ranked]
    assert scores == sorted(scores, reverse=True), "Jobs are not sorted by score!"
    
    logger.info("test_rank_jobs_end_to_end passed.")


def test_edge_case_empty_candidates():
    """
    Edge Case Validation: Ensures the API safely blocks empty candidate pools
    rather than crashing the internal ML prediction pipeline.
    """
    logger.info("Executing test_edge_case_empty_candidates...")
    jobs, _ = load_data()
    
    payload = {
        "job": jobs[0],
        "candidates": []  # Empty array
    }
    
    with TestClient(app) as client:
        response = client.post("/api/v1/portal/rank-candidates", json=payload)
        
    # Pydantic schema validation (min_length=1) will throw a 422 Unprocessable Entity
    assert response.status_code == 422
    logger.info("test_edge_case_empty_candidates passed (caught by schema validation).")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("="*60)
    logger.info("Starting Task 05 API Validation Suite")
    logger.info("="*60)
    
    # Run pytest programmatically on this file
    retcode = pytest.main(["-v", os.path.abspath(__file__)])
    
    if retcode == 0:
        logger.info("="*60)
        logger.info("SUCCESS: All API integration tests passed.")
        logger.info("="*60)
    else:
        logger.error("="*60)
        logger.error("FAILURE: One or more validation tests failed.")
        logger.error("="*60)
        sys.exit(retcode)
