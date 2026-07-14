"""
train_task03.py — Task 03: Search & Discovery Pipeline

Full end-to-end pipeline:
  1. Generate synthetic training data (600 records).
  2. Train LightGBM / GBR ranker model.
  3. Evaluate: RMSE, R², Spearman ρ.
  4. Serialise model + SHA-256 + metadata JSON.
  5. Load sample students and jobs from data/.
  6. Rank jobs for each student (Student → Jobs).
  7. Rank candidates for each job (Job → Candidates).
  8. Save logs/task03_rankings.json + logs/task03_metrics.json.
  9. Generate logs/task03_ranking_heatmap.png.
 10. Print ranked tables to console.

Standing instructions: robust error handling, structured logging.
"""

import json
import os
import sys
import logging
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ---------------------------------------------------------------------------
# Logging — must be configured before importing sibling modules that log
# ---------------------------------------------------------------------------
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

import io
_stream_handler = logging.StreamHandler(stream=io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace'))
_stream_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/task03.log", encoding='utf-8'),
        _stream_handler,
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from src.model_schemas import StudentFeatures, JobFeatures
from src.data_generator import generate_synthetic_data
from src.ranker import (
    train_ranker, save_ranker, load_ranker,
    rank_jobs_for_student, rank_candidates_for_job,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_json(filepath: str):
    if not os.path.exists(filepath):
        logger.critical(f"Required data file not found: {filepath}")
        sys.exit(1)
    with open(filepath, "r") as f:
        return json.load(f)


def print_ranked_table(title: str, ranked_items, score_label: str = "Score"):
    """Pretty-print a ranked list to the console."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  {'Rank':<6} {'ID':<12} {score_label:<10}")
    print(f"  {'----':<6} {'------------':<12} {'----------':<10}")
    for rank, item in enumerate(ranked_items, start=1):
        # Strip any unicode emoji from explanation for cp1252 terminals
        print(f"  {rank:<6} {item.id:<12} {item.score:.4f}")
        for line in item.explanation:
            safe_line = line.encode('ascii', errors='replace').decode('ascii')
            print(f"         {safe_line}")
    print()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_pipeline():
    logger.info("=" * 70)
    logger.info("Starting Task 03 Pipeline: AI-Powered Search & Discovery")
    logger.info("=" * 70)

    # ── Step 1: Generate Synthetic Training Data ────────────────────────────
    logger.info("[Step 1/8] Generating synthetic training data...")
    train_df = generate_synthetic_data(n_samples=600, random_state=42)
    logger.info(f"  -> {len(train_df)} training records ready.")

    # ── Step 2: Train the Ranker Model ──────────────────────────────────────
    logger.info("[Step 2/8] Training LightGBM ranker model...")
    model, metrics = train_ranker(train_df)

    # ── Step 3: Save Metrics ─────────────────────────────────────────────────
    logger.info("[Step 3/8] Saving evaluation metrics...")
    metrics_path = "logs/task03_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"  -> Metrics saved to {metrics_path}")
    logger.info(
        f"  RMSE={metrics['rmse']} | R²={metrics['r2']} | "
        f"Spearman ρ={metrics['spearman_rho']}"
    )

    # ── Step 4: Serialise Model ───────────────────────────────────────────────
    logger.info("[Step 4/8] Serialising model artifact...")
    pkl_path = save_ranker(model, metrics)

    # ── Step 4b: Reload and Verify ───────────────────────────────────────────
    logger.info("  Verifying integrity: reloading model from disk...")
    model = load_ranker(pkl_path)
    logger.info("  Integrity check passed.")

    # ── Step 5: Load Sample Data ─────────────────────────────────────────────
    logger.info("[Step 5/8] Loading sample students and jobs...")
    try:
        students_data = load_json("data/sample_students.json")
        jobs_data     = load_json("data/sample_jobs.json")
        students = [StudentFeatures(**s) for s in students_data]
        jobs     = [JobFeatures(**j)     for j in jobs_data]
    except Exception as e:
        logger.critical(f"Data loading / validation failed: {e}", exc_info=True)
        sys.exit(1)

    logger.info(f"  -> {len(students)} students, {len(jobs)} jobs loaded.")

    # ── Step 6: Rank Jobs for Each Student ───────────────────────────────────
    logger.info("[Step 6/8] Ranking jobs for each student...")
    student_rankings = []
    for student in students:
        try:
            response = rank_jobs_for_student(model, student, jobs)
            student_rankings.append(response.model_dump())
            print_ranked_table(
                f"Jobs Ranked for Student: {student.student_id}",
                response.ranked_jobs,
                score_label="AI Score",
            )
        except Exception as e:
            logger.error(f"  Failed for student {student.student_id}: {e}", exc_info=True)

    # ── Step 7: Rank Candidates for Each Job ─────────────────────────────────
    logger.info("[Step 7/8] Ranking candidates for each job...")
    job_rankings = []
    for job in jobs:
        try:
            response = rank_candidates_for_job(model, job, students)
            job_rankings.append(response.model_dump())
            print_ranked_table(
                f"Candidates Ranked for Job: {job.job_id}",
                response.ranked_candidates,
                score_label="AI Score",
            )
        except Exception as e:
            logger.error(f"  Failed for job {job.job_id}: {e}", exc_info=True)

    # ── Step 8a: Save Rankings JSON ──────────────────────────────────────────
    rankings_path = "logs/task03_rankings.json"
    rankings_output = {
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "model_artifact": pkl_path,
        "metrics": metrics,
        "student_rankings": student_rankings,
        "job_rankings": job_rankings,
    }
    with open(rankings_path, "w") as f:
        json.dump(rankings_output, f, indent=2)
    logger.info(f"  -> Rankings saved to {rankings_path}")

    # ── Step 8b: Heatmap Visualisation ───────────────────────────────────────
    logger.info("[Step 8/8] Generating ranking score heatmap...")
    try:
        _generate_heatmap(model, students, jobs)
    except Exception as e:
        logger.error(f"  Visualisation failed (non-fatal): {e}", exc_info=True)

    logger.info("=" * 70)
    logger.info("Task 03 Pipeline completed successfully.")
    logger.info("=" * 70)


def _generate_heatmap(model, students, jobs):
    """Generate a heatmap of AI scores: rows=students, cols=jobs."""
    from src.match_vectors import compute_match_vector
    from src.ranker import score_pair

    student_ids = [s.student_id for s in students]
    job_ids     = [j.job_id     for j in jobs]

    score_matrix = np.zeros((len(students), len(jobs)))
    for i, student in enumerate(students):
        for k, job in enumerate(jobs):
            try:
                mv = compute_match_vector(student, job)
                score, _ = score_pair(model, mv)
                score_matrix[i][k] = score
            except Exception:
                score_matrix[i][k] = 0.0

    df_heat = pd.DataFrame(score_matrix, index=student_ids, columns=job_ids)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(
        df_heat,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(
        "AI Ranking Score Matrix — Student × Job\n(Task 03: Search & Discovery)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Job ID", fontsize=11)
    ax.set_ylabel("Student ID", fontsize=11)
    plt.tight_layout()

    heatmap_path = "logs/task03_ranking_heatmap.png"
    plt.savefig(heatmap_path, dpi=150)
    plt.close()
    logger.info(f"  → Heatmap saved to {heatmap_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        run_pipeline()
    except FileNotFoundError as e:
        logger.critical(f"Missing required file: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Unhandled fatal error: {e}", exc_info=True)
        sys.exit(1)
