"""
monitor.py — Live Production Monitoring for PlaceMux ML Models

Implements a thread-safe singleton for tracking live API telemetry,
including latency, request volumes, error rates, and prediction distributions.
"""

import threading
import time
import logging
import datetime

logger = logging.getLogger(__name__)

class ModelMonitor:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ModelMonitor, cls).__new__(cls)
                    cls._instance._init_state()
        return cls._instance

    def _init_state(self):
        self.metrics_lock = threading.Lock()
        self.reset()

    def reset(self):
        with self.metrics_lock:
            self.total_requests = 0
            self.successful_requests = 0
            self.failed_requests = 0
            self.total_latency_ms = 0.0
            
            # Score distribution tracking (for monitoring drift in predictions)
            self.total_scores_sum = 0.0
            self.total_scores_count = 0
            
            self.start_time = datetime.datetime.utcnow()

    def record_request(self, latency_ms: float, success: bool = True):
        """Records a single API request and its latency."""
        with self.metrics_lock:
            self.total_requests += 1
            self.total_latency_ms += latency_ms
            if success:
                self.successful_requests += 1
            else:
                self.failed_requests += 1

    def record_prediction_scores(self, scores: list):
        """Records prediction scores to monitor the live output distribution."""
        if not scores:
            return
        with self.metrics_lock:
            self.total_scores_sum += sum(scores)
            self.total_scores_count += len(scores)

    def get_metrics(self) -> dict:
        """Retrieves a snapshot of the current live metrics."""
        with self.metrics_lock:
            avg_latency = 0.0
            if self.total_requests > 0:
                avg_latency = self.total_latency_ms / self.total_requests
                
            error_rate = 0.0
            if self.total_requests > 0:
                error_rate = self.failed_requests / self.total_requests
                
            avg_score = 0.0
            if self.total_scores_count > 0:
                avg_score = self.total_scores_sum / self.total_scores_count
                
            uptime = (datetime.datetime.utcnow() - self.start_time).total_seconds()
            throughput = self.total_requests / uptime if uptime > 0 else 0.0

            return {
                "uptime_seconds": round(uptime, 2),
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "error_rate": round(error_rate, 4),
                "avg_latency_ms": round(avg_latency, 2),
                "throughput_req_per_sec": round(throughput, 2),
                "avg_prediction_score": round(avg_score, 4),
                "total_predictions_made": self.total_scores_count,
            }

# Global singleton instance
monitor = ModelMonitor()
