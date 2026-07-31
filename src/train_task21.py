"""
train_task21.py — DPDP Consent & Security Foundations (Fairness/Bias Audit)

This script performs a baseline fairness audit on the AI ranking engine.
Since real demographic data is not collected (for DPDP compliance), it
synthetically assigns demographic groups to the cohort and calculates the
Disparate Impact Ratio (DIR) based on top-5 selection rates.

Standing Instructions applied:
- NumPy style docstrings
- Defensive error handling
- Robust structured logging
- random_state=42 reproducibility
- Explicit Edge Case & Error Handling in Prediction Models
"""

import os
import sys
import json
import logging
import random
import numpy as np
from collections import defaultdict
from typing import List, Dict, Any, Tuple

# ---------------------------------------------------------------------------
# Logging & Setup
# ---------------------------------------------------------------------------
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/task21.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Reproducibility
np.random.seed(42)
random.seed(42)

def load_data() -> Tuple[List[Any], List[Any]]:
    """
    Loads real-world sample data and converts them to Pydantic schemas.
    
    Returns
    -------
    tuple
        (students, jobs) as lists of Pydantic objects.
    """
    from src.model_schemas import StudentFeatures, JobFeatures
    
    if not os.path.exists("data/sample_students.json") or not os.path.exists("data/sample_jobs.json"):
        raise FileNotFoundError("Missing sample data. Please run fetch_real_world_data.py.")
        
    with open("data/sample_students.json", "r") as f:
        student_data = json.load(f)
    with open("data/sample_jobs.json", "r") as f:
        job_data = json.load(f)
        
    students = [StudentFeatures(**s) for s in student_data[:50]]
    jobs = [JobFeatures(**j) for j in job_data[:20]]
    
    logger.info(f"Loaded {len(students)} students and {len(jobs)} jobs for the fairness audit.")
    return students, jobs

def load_model() -> Any:
    """
    Load the latest ranker model from models/.

    Returns
    -------
    model : Any
        Loaded scikit-learn / LightGBM model object.
    """
    from src.ranker import load_ranker

    models_dir = "models"
    if not os.path.exists(models_dir):
        raise FileNotFoundError("models/ directory not found.")

    pkl_files = [
        f for f in os.listdir(models_dir)
        if f.startswith("ranker_") and f.endswith(".pkl")
    ]
    if not pkl_files:
        raise FileNotFoundError("No ranker model pkl found in models/.")

    pkl_files.sort(reverse=True)
    chosen = pkl_files[0]
    model_path = os.path.join(models_dir, chosen)

    logger.info(f"Loading ranker model: {model_path}")
    return load_ranker(model_path)


def assign_mock_demographics(students: List[Any]) -> Dict[str, str]:
    """
    Assigns synthetic demographic groups to students for the purpose of the audit.
    
    Simulates a scenario where ~30% are 'Minority' and ~70% are 'Majority'.
    To make the audit realistic, we slightly correlate the minority group 
    with lower education levels to see if the model amplifies this bias.

    Parameters
    ----------
    students : List[StudentFeatures]
        The cohort of students.

    Returns
    -------
    demographics : Dict[str, str]
        Mapping from student_id to demographic group ('Majority' or 'Minority').
    """
    demographics = {}
    for student in students:
        # Simple probabilistic assignment: lower education -> slightly higher chance of being 'Minority'
        base_prob = 0.30
        if student.education_level <= 2:
            prob_minority = base_prob + 0.10
        else:
            prob_minority = base_prob - 0.10
            
        group = "Minority" if random.random() < prob_minority else "Majority"
        demographics[student.student_id] = group
        
    return demographics


def run_fairness_audit():
    """
    Executes the Task 21 end-to-end fairness and bias audit pipeline.
    Calculates the Disparate Impact Ratio (DIR) for top-5 recommendations.
    """
    from src.recommender import recommend_jobs_for_cohort
    
    logger.info("--- Starting Task 21: Fairness/Bias Audit ---")
    
    # 1. Setup
    model = load_model()
    if model is None:
        raise ValueError("Cannot audit: model is uninitialized or None.")
        
    students, jobs = load_data()
    if not students or not jobs:
        raise ValueError("Empty candidate or job pool provided. Returning.")
        
    # 2. Inject Mock Demographics
    demographics = assign_mock_demographics(students)
    minority_count = sum(1 for g in demographics.values() if g == "Minority")
    majority_count = len(students) - minority_count
    
    logger.info(f"Demographics Assigned -> Majority: {majority_count}, Minority: {minority_count}")
    
    # 3. Generate Recommendations
    logger.info("Generating Top-5 Job Recommendations for cohort...")
    top_k = 5
    try:
        cohort_recs = recommend_jobs_for_cohort(model, students, jobs, top_k=top_k)
    except Exception as e:
        logger.error(f"Failed to generate recommendations: {e}", exc_info=True)
        raise
    
    if not cohort_recs:
        raise RuntimeError("Cohort recommendations returned empty.")
        
    # 4. Calculate Selection Rates (Probability of a group member being in the Top-1 for ANY job)
    # Actually, a better measure for recommendations: 
    # Average Match Score for Minority vs Average Match Score for Majority.
    # OR Selection Rate = how many unique students in each group got at least one score > 0.8?
    # Let's do Average Top-K Match Score per group, and % of group that got a "Strong Match" (Score > 0.7)
    
    group_scores = {"Majority": [], "Minority": []}
    group_strong_matches = {"Majority": 0, "Minority": 0}
    
    for rec in cohort_recs:
        student_id = rec["student_id"]
        group = demographics[student_id]
        
        top_jobs = rec["top_jobs"]
        if not top_jobs:
            continue
            
        # Isolate fault per item logic implicitly handled inside recommend_jobs_for_cohort, 
        # but we also wrap our metric collection just in case.
        try:
            # Average score of their top K recommendations
            avg_score = float(np.mean([j["score"] for j in top_jobs]))
            if np.isnan(avg_score) or np.isinf(avg_score):
                logger.warning(f"Invalid avg score ({avg_score}). Defaulting to 0.0.")
                avg_score = 0.0
                
            group_scores[group].append(avg_score)
            
            # Check if they have at least one strong match
            if any(j["score"] >= 0.7 for j in top_jobs):
                group_strong_matches[group] += 1
        except Exception as e:
            logger.error(f"Error calculating metrics for student {student_id}: {e}")
            continue

    avg_score_majority = float(np.mean(group_scores["Majority"])) if group_scores["Majority"] else 0.0
    avg_score_minority = float(np.mean(group_scores["Minority"])) if group_scores["Minority"] else 0.0
    
    # Selection Rate: P(Strong Match | Group)
    sr_majority = group_strong_matches["Majority"] / majority_count if majority_count > 0 else 0.0
    sr_minority = group_strong_matches["Minority"] / minority_count if minority_count > 0 else 0.0
    
    # Disparate Impact Ratio (DIR) = SR_minority / SR_majority
    if sr_majority == 0.0:
        dir_value = 1.0 # Avoid division by zero
    else:
        dir_value = sr_minority / sr_majority
        
    passed_audit = dir_value >= 0.8
    
    # 5. Report Results
    print("\n" + "="*80)
    print(" ⚖️  FAIRNESS & BIAS AUDIT REPORT ⚖️ ".center(80))
    print("="*80)
    
    print(f"Demographic Split: Majority ({majority_count}), Minority ({minority_count})")
    print(f"\n[Average Top-{top_k} Recommendation Score]")
    print(f" - Majority: {avg_score_majority:.3f}")
    print(f" - Minority: {avg_score_minority:.3f}")
    print(f" - Score Difference: {abs(avg_score_majority - avg_score_minority):.3f}")
    
    print(f"\n[Selection Rate (Probability of receiving a score >= 0.7)]")
    print(f" - Majority: {sr_majority:.1%}")
    print(f" - Minority: {sr_minority:.1%}")
    
    print(f"\n[Disparate Impact Ratio (DIR)]")
    print(f" - Ratio: {dir_value:.3f} (Standard threshold is >= 0.8)")
    
    if passed_audit:
        print("\n✅ [PASS] The engine exhibits acceptable fairness based on the 80% rule.")
    else:
        print("\n❌ [FAIL] The engine exhibits potential adverse impact (DIR < 0.8).")
        
    print("="*80)
    
    # 6. Save Metrics
    metrics_path = "logs/task21_fairness_audit.json"
    output = {
        "task": "Task 21 - DPDP Consent & Security Foundations (Fairness Audit)",
        "audit_passed": passed_audit,
        "metrics": {
            "majority_count": majority_count,
            "minority_count": minority_count,
            "avg_score_majority": avg_score_majority,
            "avg_score_minority": avg_score_minority,
            "selection_rate_majority": sr_majority,
            "selection_rate_minority": sr_minority,
            "disparate_impact_ratio": dir_value
        }
    }
    with open(metrics_path, "w") as f:
        json.dump(output, f, indent=4)
    logger.info(f"Fairness audit metrics saved -> {metrics_path}")
    logger.info("--- Task 21 Fairness Audit Complete ---")


def main():
    """
    Main entry point with fatal error trap.
    """
    try:
        run_fairness_audit()
    except Exception as e:
        logger.critical(f"Unhandled fatal error in fairness audit pipeline: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
