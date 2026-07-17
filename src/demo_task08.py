"""
demo_task08.py — Task 08: Receipts, Refunds & Reconciliation
Demonstrates the Spend-Quality Guardrail in action.
"""

import sys
import logging
from src.ranker import load_ranker, rank_jobs_for_student
from src.model_schemas import StudentFeatures, JobFeatures

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def run_demo():
    print("\n" + "="*65)
    print("  TASK 08: SPEND-QUALITY GUARDRAIL DEMO")
    print("="*65 + "\n")
    
    try:
        # 1. Load the tuned v2 model from Task 07
        import os
        pkl_files = sorted(
            [f for f in os.listdir("models") if f.startswith("ranker_v2_") and f.endswith(".pkl")],
            reverse=True
        )
        if not pkl_files:
            logger.error("No v2 model found. Please run Task 07 first.")
            sys.exit(1)
            
        model_path = os.path.join("models", pkl_files[0])
        model = load_ranker(model_path)
        
        # 2. Create a dummy student
        student = StudentFeatures(
            student_id="STU-DEMO-01",
            skills_hard=["Python", "SQL"],
            skills_soft=["Communication"],
            years_experience=1.0,
            education_level=2,
            expected_salary=60000,
            preferred_location="New York",
            remote_preference="Hybrid",
            coding_score=0.6,
            communication_score=0.7
        )
        
        # 3. Create two jobs: one High Fit, one Low Fit
        high_fit_job = JobFeatures(
            job_id="JOB-HIGH-FIT",
            required_skills=["Python", "SQL"],
            min_experience=0.0,
            salary_min=60000,
            salary_max=80000,
            job_location="New York",
            work_model="Hybrid",
            min_coding_score=0.5,
            min_communication_score=0.5
        )
        
        low_fit_job = JobFeatures(
            job_id="JOB-LOW-FIT",
            required_skills=["Java", "C++", "Spring Boot", "AWS"],
            min_experience=5.0,
            salary_min=120000,
            salary_max=150000,
            job_location="San Francisco",
            work_model="On-site",
            min_coding_score=0.9,
            min_communication_score=0.9
        )
        
        # 4. Rank the jobs for the student
        response = rank_jobs_for_student(model, student, [high_fit_job, low_fit_job])
        
        # 5. Display the results highlighting the guardrail
        print(f"Student: {student.student_id}")
        print(f"Skills: {student.skills_hard} | Exp: {student.years_experience}yrs | Salary Req: ${student.expected_salary}")
        print("-" * 65)
        
        for rank in response.ranked_jobs:
            payload = rank.explanation_payload
            print(f"Job ID     : {rank.id}")
            print(f"Match Score: {rank.score:.4f} ({payload.confidence.upper()} confidence)")
            print(f"Summary    : {payload.summary}")
            
            if payload.low_fit_warning:
                print(f"[GUARDRAIL TRIGGERED] {payload.spend_warning_message}")
            else:
                print("[GUARDRAIL] Safe to apply.")
            print("-" * 65)
            
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        
if __name__ == "__main__":
    run_demo()
