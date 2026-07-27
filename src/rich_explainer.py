"""
rich_explainer.py — Enriched Recommendation Explanation Layer (Task 18)

Adds a richer explanation layer on top of the existing ExplanationPayload:
  - match_tier       : Excellent / Good / Fair / Poor (colour-banded)
  - missing_skills   : required skills the student lacks for this specific job
  - present_skills   : required skills the student already has
  - upskilling_recs  : actionable "Learn X to improve Y" recommendations
  - detailed_narrative: 2-3 sentence human-readable explanation
  - percentile_rank  : student's rank for this job within the cohort (0–1)

This module is a pure enrichment layer — it does NOT retrain any model.
All scores still come from ranker.py via explainer.build_explanation_payload().

Standing instructions: robust error handling, structured logging,
NumPy-style docstrings, random_state=42.
"""

import logging
from typing import Any, Dict, List, Optional

from src.model_schemas import StudentFeatures, JobFeatures, ExplanationPayload
from src.explainer import build_explanation_payload

logger = logging.getLogger(__name__)

# Match tier boundaries
TIER_EXCELLENT = 0.80
TIER_GOOD      = 0.65
TIER_FAIR      = 0.45


def get_match_tier(score: float) -> Dict[str, str]:
    """
    Return a match tier label and colour for a given score.

    Parameters
    ----------
    score : float
        AI match score in [0, 1].

    Returns
    -------
    dict
        Keys: 'label' (str), 'color' (hex str), 'emoji' (str).
    """
    if score >= TIER_EXCELLENT:
        return {"label": "Excellent", "color": "#22c55e", "emoji": "🟢"}
    elif score >= TIER_GOOD:
        return {"label": "Good",      "color": "#84cc16", "emoji": "🟡"}
    elif score >= TIER_FAIR:
        return {"label": "Fair",      "color": "#f59e0b", "emoji": "🟠"}
    else:
        return {"label": "Poor",      "color": "#ef4444", "emoji": "🔴"}


def compute_skill_gaps(
    student: StudentFeatures,
    job: JobFeatures,
) -> Dict[str, List[str]]:
    """
    Compute which required skills the student has and which are missing.

    Parameters
    ----------
    student : StudentFeatures
    job : JobFeatures

    Returns
    -------
    dict
        Keys: 'present' (list of skills the student has),
              'missing' (list of required skills not in student profile).
    """
    student_skills = {s.lower() for s in student.skills_hard}
    present, missing = [], []
    for skill in job.required_skills:
        if skill.lower() in student_skills:
            present.append(skill)
        else:
            missing.append(skill)
    return {"present": present, "missing": missing}


def build_upskilling_recommendations(
    missing_skills: List[str],
    job_id: str,
    all_jobs_missing: Optional[Dict[str, List[str]]] = None,
) -> List[str]:
    """
    Generate actionable upskilling recommendations for a student.

    For each missing skill, mentions how many jobs in the pool it unlocks.

    Parameters
    ----------
    missing_skills : List[str]
        Skills missing for this specific job.
    job_id : str
        The job this student is being evaluated against.
    all_jobs_missing : dict, optional
        Map of {skill: [job_ids that need it]} for cross-job upskilling hints.

    Returns
    -------
    List[str]
        Action items, one per missing skill.
    """
    recs = []
    for skill in missing_skills[:5]:          # cap at 5 to keep it scannable
        if all_jobs_missing and skill.lower() in all_jobs_missing:
            count = len(all_jobs_missing[skill.lower()])
            recs.append(
                f"Add '{skill}' to unlock {count} additional job(s) in the pool."
            )
        else:
            recs.append(
                f"Add '{skill}' to meet the mandatory requirement for {job_id}."
            )
    if not missing_skills:
        recs.append("All required skills are met — focus on preferred skills to boost your score.")
    return recs


def build_detailed_narrative(
    student: StudentFeatures,
    job: JobFeatures,
    score: float,
    tier: Dict[str, str],
    payload: ExplanationPayload,
    skill_gaps: Dict[str, List[str]],
) -> str:
    """
    Generate a 2–3 sentence plain-English explanation of the match.

    Parameters
    ----------
    student : StudentFeatures
    job : JobFeatures
    score : float
    tier : dict  (from get_match_tier)
    payload : ExplanationPayload
    skill_gaps : dict  (from compute_skill_gaps)

    Returns
    -------
    str
        Multi-sentence narrative.
    """
    try:
        # Sentence 1 — overall verdict
        s1 = (
            f"{student.student_id} is a {tier['label'].lower()} match for {job.job_id} "
            f"with an AI relevance score of {score:.0%} ({payload.confidence} confidence)."
        )

        # Sentence 2 — top strengths
        if payload.strengths:
            top2 = payload.strengths[:2]
            labels = " and ".join(s.label for s in top2)
            s2 = f"Key strengths driving this score are {labels}."
        elif payload.neutral:
            s2 = "The match is balanced across most dimensions with no dominant strengths."
        else:
            s2 = "The profile does not meet most matching criteria for this role."

        # Sentence 3 — gaps / action
        if skill_gaps["missing"]:
            gaps = ", ".join(skill_gaps["missing"][:3])
            s3 = (
                f"Missing required skill(s): {gaps} — "
                f"upskilling in these areas would materially improve the match."
            )
        elif payload.weaknesses:
            weak = payload.weaknesses[0].label
            s3 = f"The primary area for improvement is {weak}."
        else:
            s3 = "This candidate meets all requirements and is recommended for shortlisting."

        return f"{s1} {s2} {s3}"

    except Exception as e:
        logger.error(f"Narrative generation failed: {e}", exc_info=True)
        return f"Match score: {score:.0%}. See feature breakdown for details."


def enrich_recommendation(
    model: Any,
    student: StudentFeatures,
    job: JobFeatures,
    score: float,
    feature_row: Dict,
    rank_in_cohort: int,
    cohort_size: int,
    all_jobs_missing: Optional[Dict[str, List[str]]] = None,
) -> Dict:
    """
    Build a fully enriched explanation for one (student, job) pair.

    Parameters
    ----------
    model : Any
        Loaded ranker model for feature importance extraction.
    student : StudentFeatures
    job : JobFeatures
    score : float
        AI match score [0, 1].
    feature_row : dict
        Raw feature values from match_vector_to_feature_row().
    rank_in_cohort : int
        This student's rank for this job among all cohort students (1-indexed).
    cohort_size : int
        Total number of students evaluated for this job.
    all_jobs_missing : dict, optional
        {skill: [job_ids]} for cross-job upskilling hints.

    Returns
    -------
    dict
        Enriched explanation with all layers.
    """
    if model is None:
        raise ValueError("Cannot enrich: model is None.")
    if score is None or not (0.0 <= score <= 1.0):
        raise ValueError(f"Invalid score: {score}")

    try:
        # --- Core ExplanationPayload (existing engine) ---
        payload = build_explanation_payload(model, feature_row, score)

        # --- New enrichment layers ---
        tier       = get_match_tier(score)
        skill_gaps = compute_skill_gaps(student, job)
        upskilling = build_upskilling_recommendations(
            skill_gaps["missing"], job.job_id, all_jobs_missing
        )
        narrative  = build_detailed_narrative(
            student, job, score, tier, payload, skill_gaps
        )
        percentile = round(1.0 - (rank_in_cohort - 1) / max(cohort_size - 1, 1), 3)

        return {
            # Identifiers
            "student_id":          student.student_id,
            "job_id":              job.job_id,
            # Score & tier
            "score":               round(score, 4),
            "match_tier":          tier["label"],
            "match_tier_color":    tier["color"],
            "percentile_in_cohort": percentile,
            "rank_in_cohort":      rank_in_cohort,
            "cohort_size":         cohort_size,
            # Skill analysis
            "present_skills":      skill_gaps["present"],
            "missing_skills":      skill_gaps["missing"],
            "skill_coverage_pct":  round(
                len(skill_gaps["present"]) / max(len(job.required_skills), 1), 3
            ),
            # Explanation layers
            "detailed_narrative":  narrative,
            "upskilling_recs":     upskilling,
            "summary":             payload.summary,
            "confidence":          payload.confidence,
            "shortlist":           payload.shortlist,
            "shortlist_reason":    payload.shortlist_reason,
            "low_fit_warning":     payload.low_fit_warning,
            # Feature breakdown
            "strengths":  [
                {"label": c.label, "value": c.value, "contribution": c.contribution}
                for c in payload.strengths
            ],
            "weaknesses": [
                {"label": c.label, "value": c.value, "contribution": c.contribution}
                for c in payload.weaknesses
            ],
            "neutral": [
                {"label": c.label, "value": c.value, "contribution": c.contribution}
                for c in payload.neutral
            ],
            "feature_contributions": [
                {
                    "feature": c.feature,
                    "label":   c.label,
                    "value":   c.value,
                    "contribution": c.contribution,
                    "verdict": c.verdict,
                }
                for c in payload.feature_contributions
            ],
            "factors_met":   payload.factors_met,
            "factors_total": payload.factors_total,
        }

    except Exception as e:
        logger.error(
            f"enrich_recommendation failed [{student.student_id} x {job.job_id}]: {e}",
            exc_info=True,
        )
        raise
