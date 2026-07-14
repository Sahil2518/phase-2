"""
explainer.py — Match Explainability Engine for Task 04

Produces structured ExplanationPayload objects for every (student, job) match.
Uses model-native feature importances combined with actual feature values to
compute per-feature contribution scores, human-readable summaries, confidence
bands, and shortlisting recommendations.

Standing instructions: robust error handling, structured logging.
"""

import logging
import numpy as np
from typing import Any, Dict, List, Tuple

from src.model_schemas import (
    FeatureContribution,
    ExplanationPayload,
)
from src.data_generator import FEATURE_COLS

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Human-readable labels & thresholds
# ---------------------------------------------------------------------------
FEATURE_LABELS: Dict[str, str] = {
    "skill_overlap_ratio":          "Skill Match",
    "norm_experience_gap":          "Experience Fit",
    "capped_salary_ratio":          "Salary Alignment",
    "education_met":                "Education Requirement",
    "coding_threshold_met":         "Coding Proficiency",
    "communication_threshold_met":  "Communication Proficiency",
    "location_match":               "Location Compatibility",
}

# Thresholds for categorising a feature as strength / weakness / neutral
STRENGTH_THRESHOLD = 0.60   # feature value >= this → strength
WEAKNESS_THRESHOLD = 0.40   # feature value <  this → weakness
# Between 0.40 and 0.60 → neutral

# Shortlisting configuration
SHORTLIST_MIN_SCORE = 0.65
SHORTLIST_MAX_CRITICAL_FAILURES = 0   # no critical (binary) failures allowed

# Binary (threshold) features — used for confidence calculation
BINARY_FEATURES = [
    "education_met",
    "coding_threshold_met",
    "communication_threshold_met",
    "location_match",
]


# ---------------------------------------------------------------------------
# Feature importance extraction
# ---------------------------------------------------------------------------
def _extract_feature_importances(model: Any) -> np.ndarray:
    """
    Extract feature importances from the fitted model.

    Supports LightGBM, GradientBoosting, and RandomForest estimators.
    Falls back to uniform weights if the model type is unknown.
    """
    try:
        if hasattr(model, "feature_importances_"):
            importances = np.array(model.feature_importances_, dtype=float)
            # Normalise to sum to 1.0
            total = importances.sum()
            if total > 0:
                importances = importances / total
            return importances
        else:
            logger.warning(
                "Model does not expose feature_importances_. "
                "Falling back to uniform weights."
            )
            n = len(FEATURE_COLS)
            return np.ones(n) / n
    except Exception as e:
        logger.error(f"Failed to extract feature importances: {e}", exc_info=True)
        n = len(FEATURE_COLS)
        return np.ones(n) / n


# ---------------------------------------------------------------------------
# Contribution computation
# ---------------------------------------------------------------------------
def _compute_contributions(
    feature_row: Dict[str, float],
    importances: np.ndarray,
) -> List[FeatureContribution]:
    """
    Compute per-feature contribution scores.

    contribution_i = importance_i × value_i  (then re-normalised)
    """
    try:
        raw_contributions = []
        for idx, feat in enumerate(FEATURE_COLS):
            value = float(feature_row[feat])
            importance = float(importances[idx])
            raw_contrib = importance * value
            raw_contributions.append((feat, value, importance, raw_contrib))

        # Normalise contributions to sum to 1.0
        total_contrib = sum(abs(rc[3]) for rc in raw_contributions)
        if total_contrib == 0:
            total_contrib = 1.0  # avoid division by zero

        contributions = []
        for feat, value, importance, raw_contrib in raw_contributions:
            norm_contrib = raw_contrib / total_contrib

            # Categorise verdict
            if value >= STRENGTH_THRESHOLD:
                verdict = "strength"
            elif value < WEAKNESS_THRESHOLD:
                verdict = "weakness"
            else:
                verdict = "neutral"

            contributions.append(FeatureContribution(
                feature=feat,
                label=FEATURE_LABELS.get(feat, feat),
                value=round(value, 4),
                contribution=round(norm_contrib, 4),
                verdict=verdict,
            ))

        # Sort by |contribution| descending
        contributions.sort(key=lambda c: abs(c.contribution), reverse=True)
        return contributions

    except Exception as e:
        logger.error(f"Contribution computation failed: {e}", exc_info=True)
        raise


# ---------------------------------------------------------------------------
# Confidence band
# ---------------------------------------------------------------------------
def _compute_confidence(feature_row: Dict[str, float]) -> Tuple[str, int, int]:
    """
    Compute a confidence band based on how many binary threshold features
    the candidate meets.

    Returns
    -------
    confidence : 'high', 'medium', or 'low'
    factors_met : int
    factors_total : int
    """
    try:
        factors_total = len(BINARY_FEATURES)
        factors_met = sum(
            1 for feat in BINARY_FEATURES
            if float(feature_row[feat]) >= 0.5  # binary: 0 or 1
        )

        ratio = factors_met / factors_total if factors_total > 0 else 0
        if ratio >= 0.75:
            confidence = "high"
        elif ratio >= 0.50:
            confidence = "medium"
        else:
            confidence = "low"

        return confidence, factors_met, factors_total

    except Exception as e:
        logger.error(f"Confidence computation failed: {e}", exc_info=True)
        return "low", 0, len(BINARY_FEATURES)


# ---------------------------------------------------------------------------
# Shortlisting logic
# ---------------------------------------------------------------------------
def _evaluate_shortlist(
    score: float,
    feature_row: Dict[str, float],
    factors_met: int,
    factors_total: int,
) -> Tuple[bool, str]:
    """
    Determine if the match should be shortlisted.

    Rules:
      1. Score must be >= SHORTLIST_MIN_SCORE (0.65)
      2. No critical binary failures allowed (all binary features must be met)
    """
    try:
        critical_failures = factors_total - factors_met

        if score < SHORTLIST_MIN_SCORE:
            return False, (
                f"Score ({score:.2f}) below minimum threshold "
                f"({SHORTLIST_MIN_SCORE:.2f})"
            )

        if critical_failures > SHORTLIST_MAX_CRITICAL_FAILURES:
            failed_features = [
                FEATURE_LABELS.get(feat, feat)
                for feat in BINARY_FEATURES
                if float(feature_row[feat]) < 0.5
            ]
            return False, (
                f"{critical_failures} critical requirement(s) not met: "
                + ", ".join(failed_features)
            )

        return True, (
            f"Strong match (score={score:.2f}) with all {factors_total} "
            f"requirements satisfied"
        )

    except Exception as e:
        logger.error(f"Shortlist evaluation failed: {e}", exc_info=True)
        return False, "Evaluation error — defaulting to not shortlisted"


# ---------------------------------------------------------------------------
# Natural-language summary builder
# ---------------------------------------------------------------------------
def _build_summary(
    score: float,
    contributions: List[FeatureContribution],
    confidence: str,
    shortlist: bool,
) -> str:
    """Generate a one-sentence natural-language explanation of the match."""
    try:
        # Identify top 2 strengths and top weakness
        strengths = [c for c in contributions if c.verdict == "strength"]
        weaknesses = [c for c in contributions if c.verdict == "weakness"]

        # Score tier
        if score >= 0.80:
            tier = "Excellent"
        elif score >= 0.65:
            tier = "Strong"
        elif score >= 0.45:
            tier = "Moderate"
        else:
            tier = "Weak"

        # Build sentence
        parts = [f"{tier} match (score: {score:.0%})"]

        if strengths:
            top_str = strengths[:2]
            strength_text = " and ".join(
                f"{s.label} ({s.contribution:.0%})" for s in top_str
            )
            parts.append(f"driven by {strength_text}")

        if weaknesses:
            top_weak = weaknesses[0]
            parts.append(f"held back by {top_weak.label}")

        summary = ", ".join(parts) + "."

        return summary

    except Exception as e:
        logger.error(f"Summary generation failed: {e}", exc_info=True)
        return f"Match score: {score:.0%} (summary generation error)"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def build_explanation_payload(
    model: Any,
    feature_row: Dict[str, float],
    score: float,
) -> ExplanationPayload:
    """
    Build a complete ExplanationPayload for a single (student, job) match.

    Parameters
    ----------
    model : fitted ML estimator (must expose feature_importances_)
    feature_row : dict of feature name → value (from match_vector_to_feature_row)
    score : float, the predicted match score [0,1]

    Returns
    -------
    ExplanationPayload
    """
    try:
        # 1. Extract importances and compute contributions
        importances = _extract_feature_importances(model)
        contributions = _compute_contributions(feature_row, importances)

        # 2. Categorise
        strengths  = [c for c in contributions if c.verdict == "strength"]
        weaknesses = [c for c in contributions if c.verdict == "weakness"]
        neutral    = [c for c in contributions if c.verdict == "neutral"]

        # 3. Confidence band
        confidence, factors_met, factors_total = _compute_confidence(feature_row)

        # 4. Shortlisting
        shortlist, shortlist_reason = _evaluate_shortlist(
            score, feature_row, factors_met, factors_total
        )

        # 5. NL summary
        summary = _build_summary(score, contributions, confidence, shortlist)

        return ExplanationPayload(
            summary=summary,
            strengths=strengths,
            weaknesses=weaknesses,
            neutral=neutral,
            feature_contributions=contributions,
            confidence=confidence,
            shortlist=shortlist,
            shortlist_reason=shortlist_reason,
            match_score=round(score, 4),
            factors_met=factors_met,
            factors_total=factors_total,
        )

    except Exception as e:
        logger.error(
            f"build_explanation_payload failed (score={score}): {e}",
            exc_info=True,
        )
        raise
"""
explainer.py — end of module
"""
