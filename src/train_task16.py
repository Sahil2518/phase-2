"""
train_task16.py - Task 16: College Portal & Reporting API Foundations
                  Recommendation v1 Design (Demo Pipeline)

End-to-end demonstration of Rec v1:
  1. Load the Task 07 ranker model (ranker_v2)
  2. Define a synthetic college cohort (4 students) and job pool (3 postings)
  3. Run recommend_jobs_for_cohort()  -> top 3 jobs per student
  4. Run recommend_students_for_jobs() -> top 2 candidates per job
  5. Generate the Rec v1 structured report  -> logs/task16_rec_v1_report.json
  6. Save summary metrics                   -> logs/task16_metrics.json
  7. Plot recommendation heatmap            -> logs/task16_rec_heatmap.png

Standing instructions: robust error handling, structured logging,
NumPy-style docstrings, random_state=42.
"""

import os
import sys
import json
import logging
import numpy as np
from datetime import datetime

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/task16.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_STATE   = 42
REPORT_PATH    = "logs/task16_rec_v1_report.json"
METRICS_PATH   = "logs/task16_metrics.json"
HEATMAP_PATH   = "logs/task16_rec_heatmap.png"
TOP_K_JOBS     = 3   # top jobs per student
TOP_K_CANDS    = 2   # top candidates per job

# ---------------------------------------------------------------------------
# Synthetic College Cohort (4 students)
# ---------------------------------------------------------------------------
# These profiles are intentionally varied to showcase how Rec v1 handles
# different skill levels, education, and experience within one cohort.

COHORT_STUDENTS = [
    {
        "student_id":           "STU-COL-001",
        "skills_hard":          ["python", "tensorflow", "sql", "pandas", "numpy", "docker", "git", "aws"],
        "skills_soft":          ["communication", "problem solving", "collaboration"],
        "years_experience":     4.0,
        "education_level":      3,    # M.Sc
        "expected_salary":      120000.0,
        "preferred_location":   "Bangalore",
        "remote_preference":    "Hybrid",
        "coding_score":         0.85,
        "communication_score":  0.80,
    },
    {
        "student_id":           "STU-COL-002",
        "skills_hard":          ["javascript", "react", "node.js", "mongodb", "rest api", "git", "docker"],
        "skills_soft":          ["teamwork", "attention to detail", "time management"],
        "years_experience":     2.0,
        "education_level":      2,    # B.E.
        "expected_salary":      85000.0,
        "preferred_location":   "Remote",
        "remote_preference":    "Remote",
        "coding_score":         0.72,
        "communication_score":  0.75,
    },
    {
        "student_id":           "STU-COL-003",
        "skills_hard":          ["kotlin", "java", "android", "jetpack compose", "rest api", "git"],
        "skills_soft":          ["adaptability", "creativity", "presentation"],
        "years_experience":     3.0,
        "education_level":      2,    # B.Tech
        "expected_salary":      90000.0,
        "preferred_location":   "Mumbai",
        "remote_preference":    "On-site",
        "coding_score":         0.78,
        "communication_score":  0.70,
    },
    {
        "student_id":           "STU-COL-004",
        "skills_hard":          ["aws", "docker", "kubernetes", "terraform", "python", "git", "linux", "bash"],
        "skills_soft":          ["leadership", "problem solving", "communication"],
        "years_experience":     5.0,
        "education_level":      2,    # B.Sc
        "expected_salary":      130000.0,
        "preferred_location":   "Pune",
        "remote_preference":    "Hybrid",
        "coding_score":         0.82,
        "communication_score":  0.88,
    },
]

# ---------------------------------------------------------------------------
# Synthetic Job Pool (3 company partner postings)
# ---------------------------------------------------------------------------

JOB_POOL = [
    {
        "job_id":                  "JOB-COLL-001",
        "required_skills":         ["python", "tensorflow", "sql", "docker", "git", "aws"],
        "preferred_skills":        ["kubernetes", "spark"],
        "min_experience":          3.0,
        "max_experience":          None,
        "min_education":           3,    # M.Sc required
        "salary_min":              110000.0,
        "salary_max":              150000.0,
        "job_location":            "Bangalore",
        "work_model":              "Hybrid",
        "min_coding_score":        0.75,
        "min_communication_score": 0.70,
    },
    {
        "job_id":                  "JOB-COLL-002",
        "required_skills":         ["javascript", "react", "node.js", "rest api", "mongodb", "git"],
        "preferred_skills":        ["typescript", "docker", "aws"],
        "min_experience":          1.0,
        "max_experience":          None,
        "min_education":           2,    # B.E. or above
        "salary_min":              75000.0,
        "salary_max":              105000.0,
        "job_location":            "Remote",
        "work_model":              "Remote",
        "min_coding_score":        0.65,
        "min_communication_score": 0.65,
    },
    {
        "job_id":                  "JOB-COLL-003",
        "required_skills":         ["aws", "docker", "kubernetes", "terraform", "python", "git"],
        "preferred_skills":        ["azure", "jenkins", "gcp"],
        "min_experience":          4.0,
        "max_experience":          None,
        "min_education":           2,    # B.Sc or above
        "salary_min":              105000.0,
        "salary_max":              140000.0,
        "job_location":            "Pune",
        "work_model":              "Hybrid",
        "min_coding_score":        0.75,
        "min_communication_score": 0.75,
    },
]


# ---------------------------------------------------------------------------
# Step 1 — Load Model
# ---------------------------------------------------------------------------
def load_ranker_model() -> object:
    """
    Load the latest ranker model from models/ (prefers ranker_v2, falls back to ranker_v1).

    Returns
    -------
    model : Any
        Loaded scikit-learn / LightGBM model object.

    Raises
    ------
    FileNotFoundError
        If no ranker model pkl is found in models/.
    """
    from src.ranker import load_ranker

    models_dir = "models"
    if not os.path.exists(models_dir):
        raise FileNotFoundError("models/ directory not found.")

    # Prefer ranker_v2 (Task 07 conversion-optimised), fallback to ranker_v1
    pkl_files = [
        f for f in os.listdir(models_dir)
        if f.startswith("ranker_") and f.endswith(".pkl")
    ]
    if not pkl_files:
        raise FileNotFoundError("No ranker model pkl found in models/.")

    # Sort descending to prefer v2 over v1, and newest date
    pkl_files.sort(reverse=True)
    chosen = pkl_files[0]
    model_path = os.path.join(models_dir, chosen)

    logger.info(f"Loading ranker model: {model_path}")
    model = load_ranker(model_path)
    logger.info(f"Model loaded successfully.")
    return model


# ---------------------------------------------------------------------------
# Step 2 — Build Pydantic Profiles
# ---------------------------------------------------------------------------
def build_profiles() -> tuple:
    """
    Instantiate StudentFeatures and JobFeatures Pydantic objects from raw dicts.

    Returns
    -------
    tuple
        (students: List[StudentFeatures], jobs: List[JobFeatures])
    """
    from src.model_schemas import StudentFeatures, JobFeatures

    students = [StudentFeatures(**s) for s in COHORT_STUDENTS]
    jobs     = [JobFeatures(**j)     for j in JOB_POOL]

    logger.info(f"Built {len(students)} student profiles and {len(jobs)} job profiles.")
    return students, jobs


# ---------------------------------------------------------------------------
# Step 3 — Run Recommendations
# ---------------------------------------------------------------------------
def run_recommendations(model, students, jobs) -> tuple:
    """
    Run cohort recommendations and job shortlists using the Rec v1 engine.

    Parameters
    ----------
    model : Any
        Loaded ranker model.
    students : List[StudentFeatures]
    jobs : List[JobFeatures]

    Returns
    -------
    tuple
        (cohort_rec: List[Dict], job_shortlists: List[Dict])
    """
    from src.recommender import recommend_jobs_for_cohort, recommend_students_for_jobs

    logger.info("=" * 60)
    logger.info("  STEP 3a — Cohort Recommendations (jobs per student)")
    logger.info("=" * 60)
    cohort_rec = recommend_jobs_for_cohort(
        model=model,
        students=students,
        jobs=jobs,
        top_k=TOP_K_JOBS,
    )

    logger.info("=" * 60)
    logger.info("  STEP 3b — Job Shortlists (candidates per job)")
    logger.info("=" * 60)
    job_shortlists = recommend_students_for_jobs(
        model=model,
        students=students,
        jobs=jobs,
        top_k=TOP_K_CANDS,
    )

    return cohort_rec, job_shortlists


# ---------------------------------------------------------------------------
# Step 4 — Generate Report
# ---------------------------------------------------------------------------
def generate_and_save_report(students, jobs, cohort_rec, job_shortlists) -> dict:
    """
    Assemble the Rec v1 report and save to disk.

    Parameters
    ----------
    students : List[StudentFeatures]
    jobs : List[JobFeatures]
    cohort_rec : List[Dict]
    job_shortlists : List[Dict]

    Returns
    -------
    dict  Full Rec v1 report.
    """
    from src.recommender import generate_rec_report

    report = generate_rec_report(
        students=students,
        jobs=jobs,
        cohort_recommendations=cohort_rec,
        job_shortlists=job_shortlists,
    )

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Rec v1 report saved -> {REPORT_PATH}")

    # Summary metrics
    summary = report["college_summary"]
    metrics = {
        "task":                   "Task 16 - Rec v1 Design",
        "timestamp":              report["timestamp"],
        "college_cohort_size":    report["college_cohort_size"],
        "jobs_evaluated":         report["jobs_evaluated"],
        "avg_cohort_score":       summary["avg_cohort_score"],
        "placement_ready_count":  summary["placement_ready_count"],
        "placement_ready_pct":    summary["placement_ready_pct"],
        "top_skill_gap":          summary["top_skill_gap"],
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Summary metrics saved -> {METRICS_PATH}")

    return report


# ---------------------------------------------------------------------------
# Step 5 — Heatmap
# ---------------------------------------------------------------------------
def plot_heatmap(students, jobs, cohort_rec: list) -> None:
    """
    Save a heatmap of match scores: rows = students, columns = jobs.

    Each cell shows the AI match score for that (student, job) pair.
    Scores are read from the cohort_rec output; cells with no data show 0.

    Parameters
    ----------
    students : List[StudentFeatures]
    jobs : List[JobFeatures]
    cohort_rec : List[Dict]
        Output of recommend_jobs_for_cohort().
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        student_ids = [s.student_id for s in students]
        job_ids     = [j.job_id     for j in jobs]

        # Build score matrix
        score_matrix = np.zeros((len(student_ids), len(job_ids)))
        job_idx = {jid: i for i, jid in enumerate(job_ids)}

        for rec in cohort_rec:
            sid = rec["student_id"]
            row = student_ids.index(sid)
            for top_job in rec["top_jobs"]:
                col = job_idx.get(top_job["job_id"])
                if col is not None:
                    score_matrix[row][col] = top_job["score"]

        fig, ax = plt.subplots(figsize=(8, 5))
        cmap = plt.cm.RdYlGn
        im   = ax.imshow(score_matrix, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")

        # Labels
        ax.set_xticks(range(len(job_ids)))
        ax.set_xticklabels(job_ids, rotation=15, ha="right", fontsize=10)
        ax.set_yticks(range(len(student_ids)))
        ax.set_yticklabels(student_ids, fontsize=10)

        # Annotate cells with score values
        for i in range(len(student_ids)):
            for j in range(len(job_ids)):
                score = score_matrix[i][j]
                color = "white" if score < 0.45 or score > 0.72 else "black"
                ax.text(
                    j, i,
                    f"{score:.2f}" if score > 0 else "-",
                    ha="center", va="center",
                    fontsize=10, fontweight="bold", color=color,
                )

        plt.colorbar(im, ax=ax, label="Match Score")
        ax.set_title(
            "Task 16 — Rec v1 Match Score Heatmap\n"
            "College Cohort × Job Pool",
            fontsize=12, fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(HEATMAP_PATH, dpi=150)
        plt.close()
        logger.info(f"Heatmap saved -> {HEATMAP_PATH}")

    except Exception as e:
        logger.warning(f"Heatmap generation failed (non-fatal): {e}")


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def run_pipeline() -> None:
    """
    End-to-end Task 16 Rec v1 pipeline:
    1. Load ranker_v2 model
    2. Build Pydantic student + job profiles
    3. Run cohort recommendations + job shortlists
    4. Generate and save Rec v1 report + metrics
    5. Plot recommendation heatmap
    """
    logger.info("=" * 60)
    logger.info("  PlaceMux Task 16 — Rec v1 Design Demo")
    logger.info(f"  Cohort: {len(COHORT_STUDENTS)} students | Jobs: {len(JOB_POOL)}")
    logger.info("=" * 60)

    model             = load_ranker_model()
    students, jobs    = build_profiles()
    cohort_rec, jsl   = run_recommendations(model, students, jobs)

    if not cohort_rec:
        raise RuntimeError("No cohort recommendations produced — aborting.")

    report = generate_and_save_report(students, jobs, cohort_rec, jsl)
    plot_heatmap(students, jobs, cohort_rec)

    logger.info("Task 16 pipeline complete. Rec v1 design ready. ✅")


def main():
    try:
        run_pipeline()
    except FileNotFoundError as e:
        logger.critical(f"Missing required file: {e}")
        sys.exit(1)
    except RuntimeError as e:
        logger.critical(f"Runtime error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Unhandled fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
