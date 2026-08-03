"""
train_task24.py — Task 24: Fairness Close and Sign-off
Main wrapper for the fairness audit pipeline.
Follows standing instructions for robust error handling and execution.
"""

import sys
import logging
from fairness_audit import run_fairness_audit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/task24_main.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

def main():
    """
    Executes the fairness audit and model sign-off.
    """
    logger.info("Initializing Task 24 Pipeline: Launch Rehearsal (Fairness Audit)")
    try:
        run_fairness_audit()
        logger.info("Task 24 Pipeline completed successfully.")
    except Exception as e:
        logger.critical(f"Unhandled fatal error in Task 24: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
