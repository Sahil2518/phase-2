"""
train_task25.py — Task 25: Go-Live (Live Model Monitoring)

Simulates live production traffic by hitting the FastAPI endpoints via TestClient.
Collects and verifies telemetry (latency, request counts, prediction scores).
"""

import os
import sys
import json
import logging
import random
from fastapi.testclient import TestClient

from src.rec_api import app

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/task25.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def run_production_simulation():
    logger.info("=" * 60)
    logger.info("Starting Task 25: Live Model Monitoring Simulation")
    logger.info("=" * 60)

    # 1. Fetch data
    logger.info("Loading realistic sample data payloads...")
    try:
        with open("data/sample_jobs.json", "r") as f:
            jobs = json.load(f)[:10]
        with open("data/sample_students.json", "r") as f:
            students = json.load(f)[:10]
    except Exception as e:
        raise ValueError(f"Failed to load test data for Go-Live simulation: {e}")
    
    if not jobs or not students:
        raise ValueError("Failed to generate test data for Go-Live simulation.")

    # We must invoke lifespan context for TestClient to load the model
    with TestClient(app) as client:
        # Check health
        health = client.get("/health")
        if health.status_code != 200:
            raise RuntimeError(f"API Health Check Failed: {health.text}")
        logger.info("API is live. Commencing traffic simulation...")
        
        num_requests = 100
        logger.info(f"Simulating {num_requests} concurrent-style requests to endpoints...")
        
        # We will interleave requests to both endpoints
        for i in range(num_requests):
            # Endpoint 1: Jobs for Cohort
            if i % 2 == 0:
                payload = {
                    "students": random.sample(students, k=min(3, len(students))),
                    "jobs": jobs,
                    "top_k": 3
                }
                resp = client.post("/recommend/jobs-for-cohort", json=payload)
                if resp.status_code != 200:
                    logger.warning(f"Request {i} failed: {resp.text}")
            
            # Endpoint 2: Students for Job
            else:
                payload = {
                    "job": random.choice(jobs),
                    "students": students,
                    "top_k": 2
                }
                resp = client.post("/recommend/students-for-job", json=payload)
                if resp.status_code != 200:
                    logger.warning(f"Request {i} failed: {resp.text}")

        # Intentional Bad Request for Error Rate testing
        logger.info("Injecting a malformed request to test error tracking...")
        client.post("/recommend/jobs-for-cohort", json={"students": [], "jobs": [], "top_k": 3})

        # Fetch Live Metrics
        logger.info("Fetching /live-metrics telemetry...")
        metrics_resp = client.get("/live-metrics")
        if metrics_resp.status_code != 200:
            raise RuntimeError(f"Failed to fetch live-metrics: {metrics_resp.text}")
            
        metrics = metrics_resp.json()
        
        logger.info("=" * 60)
        logger.info("Live Production Telemetry Snapshot:")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v}")
        logger.info("=" * 60)
        
        # Validation
        if metrics["total_requests"] == 0:
            raise ValueError("Telemetry failure: total_requests is 0.")
        if metrics["failed_requests"] == 0:
            raise ValueError("Telemetry failure: failed_requests is 0 (expected 1 from our intentional bad request).")
        if metrics["avg_latency_ms"] <= 0:
            raise ValueError("Telemetry failure: average latency is not being tracked correctly.")
            
        # Save Snapshot
        snapshot_path = "logs/task25_live_metrics.json"
        with open(snapshot_path, "w") as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Telemetry snapshot saved to {snapshot_path}")

def main():
    try:
        run_production_simulation()
    except Exception as e:
        logger.critical(f"Unhandled fatal error in production simulation: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
