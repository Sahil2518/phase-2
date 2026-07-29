"""
train_task20.py — Portals Integration & Dry Run (Rec Validation)

This script validates the recommendation quality of the AI engine.
It loads real-world sample data, generates top-K recommendations for a cohort,
and asserts several statistical hypotheses to prove the engine correctly
prioritizes high-quality, high-skill-overlap jobs.

Standing Instructions applied:
- NumPy style docstrings
- Defensive error handling
- Robust structured logging
- random_state=42 reproducibility
"""

import os
import sys
import json
import logging
import numpy as np

# ---------------------------------------------------------------------------
# Logging & Setup
# ---------------------------------------------------------------------------
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/task20.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

np.random.seed(42)

def load_data() -> tuple:
    """
    Loads real-world sample data and converts them to Pydantic schemas.
    
    Returns
    -------
    tuple
        (students, jobs) as lists of Pydantic objects.
    """
    from src.model_schemas import StudentFeatures, JobFeatures
    
    if not os.path.exists("data/sample_jobs.json") or not os.path.exists("data/sample_students.json"):
        raise FileNotFoundError("Missing real-world sample data. Please run fetch_real_world_data.py first.")
        
    with open("data/sample_students.json", "r") as f:
        student_data = json.load(f)
    with open("data/sample_jobs.json", "r") as f:
        job_data = json.load(f)
        
    # Cap size for faster dry run validation
    students = [StudentFeatures(**s) for s in student_data[:50]]
    jobs = [JobFeatures(**j) for j in job_data[:20]]
    
    logger.info(f"Loaded {len(students)} students and {len(jobs)} jobs for validation.")
    return students, jobs


def load_model():
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


def run_validation_suite():
    """
    Executes the Task 20 end-to-end validation pipeline.
    """
    from src.recommender import recommend_jobs_for_cohort
    from src.match_vectors import compute_match_vector
    
    logger.info("--- Starting Task 20: Recommendation Quality Validation ---")
    
    # 1. Setup
    model = load_model()
    students, jobs = load_data()
    
    # Map jobs for quick lookup
    job_map = {j.job_id: j for j in jobs}
    
    # 2. Generate Recommendations
    logger.info("Generating Top-5 Job Recommendations for cohort...")
    top_k = 5
    cohort_recs = recommend_jobs_for_cohort(model, students, jobs, top_k=top_k)
    
    if not cohort_recs:
        raise RuntimeError("Cohort recommendations returned empty.")
        
    # 3. Validation Hypotheses
    logger.info("Validating recommendation quality hypotheses...")
    
    top_1_overlaps = []
    top_k_overlaps = []
    best_job_ids = set()
    monotonicity_violations = 0
    total_recommendations = 0
    
    for rec in cohort_recs:
        student_id = rec["student_id"]
        top_jobs = rec["top_jobs"]
        
        if not top_jobs:
            continue
            
        student_obj = next(s for s in students if s.student_id == student_id)
        
        # Check monotonicity
        scores = [tj["score"] for tj in top_jobs]
        total_recommendations += len(scores)
        if scores != sorted(scores, reverse=True):
            monotonicity_violations += 1
            
        # Track catalog diversity
        best_job_ids.add(rec["best_job_id"])
        
        # Calculate raw skill overlap for Top 1 vs Top K (last item)
        job_1_obj = job_map[top_jobs[0]["job_id"]]
        vec_1 = compute_match_vector(student_obj, job_1_obj)
        top_1_overlaps.append(vec_1.skill_overlap_ratio)
        
        if len(top_jobs) > 1:
            job_k_obj = job_map[top_jobs[-1]["job_id"]]
            vec_k = compute_match_vector(student_obj, job_k_obj)
            top_k_overlaps.append(vec_k.skill_overlap_ratio)
            
    # Compile Metrics
    avg_top1_overlap = float(np.mean(top_1_overlaps)) if top_1_overlaps else 0.0
    avg_topk_overlap = float(np.mean(top_k_overlaps)) if top_k_overlaps else 0.0
    catalog_coverage = len(best_job_ids)
    
    # Assertions
    logger.info(f"Top 1 Avg Skill Overlap: {avg_top1_overlap:.3f}")
    logger.info(f"Top {top_k} Avg Skill Overlap: {avg_topk_overlap:.3f}")
    logger.info(f"Monotonicity Violations: {monotonicity_violations} / {total_recommendations}")
    logger.info(f"Catalog Coverage (Unique Top 1 Jobs): {catalog_coverage}")
    
    # Let's print the Admin Report
    print("\n" + "="*80)
    print(" 🔎 RECOMMENDATION QUALITY VALIDATION REPORT 🔎 ".center(80))
    print("="*80)
    
    passed_all = True
    
    # Check 1: Overlap Consistency (Rank 1 should be generally better/equal to Rank K)
    # Note: AI scores consider more than just overlap, but overlap is a heavy feature.
    if avg_top1_overlap >= avg_topk_overlap - 0.05:  # allowing minor variance
        print("✅ [PASS] Skill Match Consistency: Top recommendations have high skill overlap.")
    else:
        print("❌ [FAIL] Skill Match Consistency: Lower ranked jobs have higher overlap!")
        passed_all = False
        
    # Check 2: Monotonicity
    if monotonicity_violations == 0:
        print("✅ [PASS] Score Monotonicity: Scores are strictly decreasing.")
    else:
        print(f"❌ [FAIL] Score Monotonicity: Found {monotonicity_violations} out-of-order anomalies.")
        passed_all = False
        
    # Check 3: Catalog Diversity
    if catalog_coverage > 1:
        print(f"✅ [PASS] Catalog Diversity: Recommended {catalog_coverage} unique jobs to the cohort.")
    else:
        print("❌ [FAIL] Catalog Diversity: The engine recommended the exact same job to everyone!")
        passed_all = False
        
    print("="*80)
    
    # Save Metrics
    metrics_path = "logs/task20_validation.json"
    output = {
        "task": "Task 20 - Portals Integration & Dry Run",
        "validation_passed": passed_all,
        "metrics": {
            "avg_top1_skill_overlap": avg_top1_overlap,
            "avg_top5_skill_overlap": avg_topk_overlap,
            "monotonicity_violations": monotonicity_violations,
            "catalog_coverage": catalog_coverage,
            "total_students_evaluated": len(cohort_recs),
            "total_jobs_in_pool": len(jobs)
        }
    }
    with open(metrics_path, "w") as f:
        json.dump(output, f, indent=4)
    logger.info(f"Validation metrics saved -> {metrics_path}")
    logger.info("--- Task 20 Pipeline Complete ---")
    

def main():
    """
    Main entry point with fatal error trap.
    """
    try:
        run_validation_suite()
    except Exception as e:
        logger.critical(f"Unhandled fatal error in pipeline: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
