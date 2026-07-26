"""
recommender.py — Recommendation v1 Engine (Task 16)

College Portal recommendation layer built on top of the Task 03/07 ranker.
Provides three public functions:

  1. recommend_jobs_for_cohort()     — top-k jobs per student in a college cohort
  2. recommend_students_for_jobs()  — top-k candidates per job posting
  3. generate_rec_report()           — assembles the full Rec v1 structured report

This module is a pure orchestration layer: it does NOT train any model.
It delegates all scoring to the existing rank_jobs_for_student() and
rank_candidates_for_job() functions from ranker.py.

Standing instructions: robust error handling, structured logging,
NumPy-style docstrings, random_state=42.
"""

import logging
import datetime
from typing import Any, List, Dict, Optional

from src.model_schemas import StudentFeatures, JobFeatures
from src.ranker import rank_jobs_for_student, rank_candidates_for_job

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# Threshold above which a student is considered "placement-ready"
PLACEMENT_READY_THRESHOLD = 0.50


# ---------------------------------------------------------------------------
# Public Function 1 — Cohort Recommendations
# ---------------------------------------------------------------------------
def recommend_jobs_for_cohort(
    model: Any,
    students: List[StudentFeatures],
    jobs: List[JobFeatures],
    top_k: int = 3,
) -> List[Dict]:
    """
    Produce ranked job recommendations for every student in the college cohort.

    For each student, calls rank_jobs_for_student() and extracts the top_k
    results. Students that fail individually are logged and skipped so one
    bad profile does not abort the entire cohort run.

    Parameters
    ----------
    model : Any
        Loaded ranker model (LightGBM or GBR) from ranker.load_ranker().
    students : List[StudentFeatures]
        The college's student cohort.
    jobs : List[JobFeatures]
        Pool of company job postings to rank against.
    top_k : int
        Maximum number of top jobs to return per student. Default 3.

    Returns
    -------
    List[Dict]
        One dict per student with keys:
        - student_id       : str
        - top_jobs         : list of {job_id, score, rank, explanation}
        - avg_match_score  : float — mean score across top_k recommendations
        - best_job_id      : str — job_id with the highest score
        - placement_ready  : bool — True if avg_match_score >= PLACEMENT_READY_THRESHOLD

    Raises
    ------
    ValueError
        If model is None, or if students / jobs lists are empty.
    """
    if model is None:
        raise ValueError("Cannot recommend: model is None.")
    if not students:
        raise ValueError("Cannot recommend: students list is empty.")
    if not jobs:
        raise ValueError("Cannot recommend: jobs list is empty.")

    logger.info(
        f"recommend_jobs_for_cohort — {len(students)} students x {len(jobs)} jobs | top_k={top_k}"
    )

    cohort_results = []

    for student in students:
        sid = student.student_id
        try:
            response = rank_jobs_for_student(
                model=model,
                student=student,
                jobs=jobs,
                top_k=top_k,
            )

            top_jobs = []
            for rank, result in enumerate(response.ranked_jobs, start=1):
                top_jobs.append({
                    "job_id":      result.id,
                    "score":       result.score,
                    "rank":        rank,
                    "explanation": result.explanation[:2] if result.explanation else [],
                })

            scores = [j["score"] for j in top_jobs]
            avg_score = round(sum(scores) / len(scores), 4) if scores else 0.0
            best_job  = top_jobs[0]["job_id"] if top_jobs else None

            cohort_results.append({
                "student_id":      sid,
                "top_jobs":        top_jobs,
                "avg_match_score": avg_score,
                "best_job_id":     best_job,
                "placement_ready": avg_score >= PLACEMENT_READY_THRESHOLD,
            })
            logger.info(
                f"  [{sid}] top job: {best_job} (score={top_jobs[0]['score'] if top_jobs else 0:.4f})"
                f" | avg: {avg_score:.4f} | placement_ready: {avg_score >= PLACEMENT_READY_THRESHOLD}"
            )

        except Exception as e:
            logger.error(f"  [{sid}] Cohort recommendation FAILED: {e}", exc_info=True)

    logger.info(
        f"Cohort recommendation complete. "
        f"{len(cohort_results)}/{len(students)} students processed."
    )
    return cohort_results


# ---------------------------------------------------------------------------
# Public Function 2 — Job Shortlists
# ---------------------------------------------------------------------------
def recommend_students_for_jobs(
    model: Any,
    students: List[StudentFeatures],
    jobs: List[JobFeatures],
    top_k: int = 2,
) -> List[Dict]:
    """
    Produce ranked candidate shortlists for every job posting.

    For each job, calls rank_candidates_for_job() and extracts the top_k
    candidates. Useful for college placement reporting to company partners.
    Fault-isolated per job.

    Parameters
    ----------
    model : Any
        Loaded ranker model.
    students : List[StudentFeatures]
        Pool of college students (candidates).
    jobs : List[JobFeatures]
        Job postings to shortlist candidates for.
    top_k : int
        Maximum number of top candidates to return per job. Default 2.

    Returns
    -------
    List[Dict]
        One dict per job with keys:
        - job_id           : str
        - top_candidates   : list of {student_id, score, rank}
        - avg_candidate_score : float

    Raises
    ------
    ValueError
        If model is None, or lists are empty.
    """
    if model is None:
        raise ValueError("Cannot shortlist: model is None.")
    if not students:
        raise ValueError("Cannot shortlist: students list is empty.")
    if not jobs:
        raise ValueError("Cannot shortlist: jobs list is empty.")

    logger.info(
        f"recommend_students_for_jobs — {len(jobs)} jobs x {len(students)} candidates | top_k={top_k}"
    )

    job_results = []

    for job in jobs:
        jid = job.job_id
        try:
            response = rank_candidates_for_job(
                model=model,
                job=job,
                students=students,
                top_k=top_k,
            )

            top_candidates = []
            for rank, result in enumerate(response.ranked_candidates, start=1):
                top_candidates.append({
                    "student_id": result.id,
                    "score":      result.score,
                    "rank":       rank,
                })

            scores = [c["score"] for c in top_candidates]
            avg_score = round(sum(scores) / len(scores), 4) if scores else 0.0

            job_results.append({
                "job_id":               jid,
                "top_candidates":       top_candidates,
                "avg_candidate_score":  avg_score,
            })
            logger.info(
                f"  [{jid}] top candidate: "
                f"{top_candidates[0]['student_id'] if top_candidates else 'none'}"
                f" (score={top_candidates[0]['score'] if top_candidates else 0:.4f})"
            )

        except Exception as e:
            logger.error(f"  [{jid}] Job shortlist FAILED: {e}", exc_info=True)

    logger.info(
        f"Job shortlist complete. {len(job_results)}/{len(jobs)} jobs processed."
    )
    return job_results


# ---------------------------------------------------------------------------
# Public Function 3 — Assemble Rec v1 Report
# ---------------------------------------------------------------------------
def generate_rec_report(
    students: List[StudentFeatures],
    jobs: List[JobFeatures],
    cohort_recommendations: List[Dict],
    job_shortlists: List[Dict],
) -> Dict:
    """
    Assemble the full Rec v1 structured report from cohort and job results.

    Computes college-level aggregate statistics:
    - Average match score across the entire cohort
    - Number and percentage of placement-ready students
    - Most common skill gap (job required skill not found in any top match)

    Parameters
    ----------
    students : List[StudentFeatures]
        The college cohort (used for skill gap analysis).
    jobs : List[JobFeatures]
        The job pool (used for skill gap analysis).
    cohort_recommendations : List[Dict]
        Output of recommend_jobs_for_cohort().
    job_shortlists : List[Dict]
        Output of recommend_students_for_jobs().

    Returns
    -------
    Dict
        Full Rec v1 report with keys: task, timestamp, college_cohort_size,
        jobs_evaluated, cohort_recommendations, job_shortlists, college_summary.
    """
    # --- College-level summary stats ---
    all_avg_scores = [r["avg_match_score"] for r in cohort_recommendations]
    avg_cohort_score = round(sum(all_avg_scores) / len(all_avg_scores), 4) if all_avg_scores else 0.0

    ready_count = sum(1 for r in cohort_recommendations if r.get("placement_ready", False))
    ready_pct   = round(ready_count / len(cohort_recommendations) * 100, 1) if cohort_recommendations else 0.0

    # --- Skill gap: required job skills not covered by any student's hard skills ---
    all_student_skills = set()
    for s in students:
        all_student_skills.update([sk.lower() for sk in s.skills_hard])

    skill_gap_counts: Dict[str, int] = {}
    for job in jobs:
        for skill in job.required_skills:
            if skill.lower() not in all_student_skills:
                skill_gap_counts[skill.lower()] = skill_gap_counts.get(skill.lower(), 0) + 1

    top_skill_gap = (
        max(skill_gap_counts, key=skill_gap_counts.get)
        if skill_gap_counts else "none"
    )

    college_summary = {
        "avg_cohort_score":       avg_cohort_score,
        "placement_ready_count":  ready_count,
        "placement_ready_pct":    ready_pct,
        "top_skill_gap":          top_skill_gap,
        "skill_gap_detail":       skill_gap_counts,
    }

    report = {
        "task":                   "Task 16 - Rec v1 Design",
        "timestamp":              datetime.datetime.now().isoformat(),
        "college_cohort_size":    len(students),
        "jobs_evaluated":         len(jobs),
        "cohort_recommendations": cohort_recommendations,
        "job_shortlists":         job_shortlists,
        "college_summary":        college_summary,
    }

    logger.info("=" * 55)
    logger.info("  REC V1 COLLEGE SUMMARY")
    logger.info("=" * 55)
    logger.info(f"  Cohort size        : {len(students)}")
    logger.info(f"  Jobs evaluated     : {len(jobs)}")
    logger.info(f"  Avg cohort score   : {avg_cohort_score:.4f}")
    logger.info(f"  Placement-ready    : {ready_count}/{len(cohort_recommendations)} ({ready_pct:.1f}%)")
    logger.info(f"  Top skill gap      : {top_skill_gap}")
    logger.info("=" * 55)

    return report
