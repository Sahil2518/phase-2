"""
match_vectors.py — Match Feature Vector Computation (Task 02)

This module is the core feature engineering layer of the PlaceMux matching pipeline.
It takes a raw (StudentFeatures, JobFeatures) pair and derives a structured MatchVector
of normalised numerical features that the ML ranking model (Task 03) consumes as input.

Architecture Note:
    This module acts as the bridge between raw profile data (Pydantic schemas) and
    the ML model's feature space. Every feature is independently bounded and guarded
    so that invalid inputs never propagate silently into the model.

Edge Cases Handled:
    - Empty job required_skills list → skill_overlap_ratio defaults to 1.0
    - Zero expected_salary → salary ratio defaults to 1.0 (avoids ZeroDivisionError)
    - Remote work model → location is always a match regardless of city
    - Any unexpected error is logged with full traceback and re-raised
"""

import logging
import os

# Ensure logs directory exists before configuring the file handler
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/task02_matching.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

from src.model_schemas import StudentFeatures, JobFeatures, MatchVector


def compute_match_vector(student: StudentFeatures, job: JobFeatures) -> MatchVector:
    """
    Derive a structured MatchVector from a (StudentFeatures, JobFeatures) pair.

    This is the primary feature engineering function. It converts raw profile
    attributes into normalised numerical signals that capture the quality of
    the match along 7 independent dimensions.

    Parameters
    ----------
    student : StudentFeatures
        The student/candidate profile validated by Pydantic.
    job : JobFeatures
        The job posting validated by Pydantic.

    Returns
    -------
    MatchVector
        A Pydantic object containing 7 derived features:
        - skill_overlap_ratio         : Fraction of job's required skills met [0, 1]
        - experience_gap              : student years − job min (can be negative)
        - salary_match_ratio          : job.salary_max / student.expected_salary
        - location_match              : 1.0 if location aligns or job is remote
        - education_met               : 1.0 if student meets minimum education
        - coding_threshold_met        : 1.0 if student meets coding score threshold
        - communication_threshold_met : 1.0 if student meets comms threshold

    Raises
    ------
    Exception
        Any unexpected error is logged with full traceback and re-raised to be
        caught by the caller's fault isolation guard in ranker.py.

    Edge Cases
    ----------
    - Empty required_skills : Returns skill_overlap_ratio = 1.0 (no bar = perfect fit).
    - Zero expected_salary  : Returns salary_match_ratio = 1.0 to prevent ZeroDivisionError.
    - Remote work_model     : location_match forced to True regardless of city comparison.
    """
    try:
        # ------------------------------------------------------------------
        # 1. Skill Overlap Ratio
        # Measures what fraction of the JOB's required skills the student has.
        # Edge case: If the job has no required skills, student trivially meets all.
        # ------------------------------------------------------------------
        student_skills = set(s.lower() for s in student.skills_hard)
        job_skills = set(s.lower() for s in job.required_skills)

        if len(job_skills) == 0:
            # Edge case: no requirements → 100% overlap by definition
            skill_overlap_ratio = 1.0
        else:
            overlap = student_skills.intersection(job_skills)
            skill_overlap_ratio = len(overlap) / len(job_skills)

        # ------------------------------------------------------------------
        # 2. Experience Gap
        # Positive = overqualified (good). Negative = underqualified.
        # The ranker normalises this, so raw gap is preserved here.
        # ------------------------------------------------------------------
        experience_gap = student.years_experience - job.min_experience

        # ------------------------------------------------------------------
        # 3. Salary Match Ratio
        # Ratio > 1.0 means the job budget exceeds the student's expectation.
        # Edge case: expected_salary = 0 would cause ZeroDivisionError.
        # ------------------------------------------------------------------
        if student.expected_salary > 0:
            salary_match_ratio = job.salary_max / student.expected_salary
        else:
            logger.warning(
                f"Student {student.student_id} has expected_salary=0. "
                "Defaulting salary_match_ratio to 1.0 to avoid ZeroDivisionError."
            )
            salary_match_ratio = 1.0

        # ------------------------------------------------------------------
        # 4. Location Match
        # True if: job is Remote (city irrelevant) OR cities match (case-insensitive).
        # ------------------------------------------------------------------
        location_match = (
            job.work_model.lower() == "remote" or
            student.preferred_location.lower() == job.job_location.lower()
        )

        # ------------------------------------------------------------------
        # 5. Education Requirement Met
        # Encoded as 1.0/0.0 boolean for model compatibility.
        # ------------------------------------------------------------------
        education_met = student.education_level >= job.min_education

        # ------------------------------------------------------------------
        # 6. Competency Threshold Checks
        # Binary: does the student meet the job's minimum proficiency bars?
        # ------------------------------------------------------------------
        coding_threshold_met = student.coding_score >= job.min_coding_score
        communication_threshold_met = (
            student.communication_score >= job.min_communication_score
        )

        return MatchVector(
            student_id=student.student_id,
            job_id=job.job_id,
            skill_overlap_ratio=skill_overlap_ratio,
            experience_gap=experience_gap,
            salary_match_ratio=salary_match_ratio,
            location_match=location_match,
            education_met=education_met,
            coding_threshold_met=coding_threshold_met,
            communication_threshold_met=communication_threshold_met,
        )

    except Exception as e:
        logger.error(
            f"Failed to compute match vector for "
            f"Student {student.student_id} x Job {job.job_id}: {e}",
            exc_info=True,
        )
        raise
