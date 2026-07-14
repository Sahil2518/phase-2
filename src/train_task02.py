import json
import os
import sys
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List

from src.model_schemas import StudentFeatures, JobFeatures
from src.match_vectors import compute_match_vector, logger

def load_json(filepath: str) -> List[dict]:
    if not os.path.exists(filepath):
        logger.critical(f"Missing required data file: {filepath}")
        sys.exit(1)
    with open(filepath, 'r') as f:
        return json.load(f)

def run_pipeline():
    logger.info("Starting Task 02 Pipeline: Match Vectors and Thresholds")
    
    # 1. Load Data
    students_data = load_json("data/sample_students.json")
    jobs_data = load_json("data/sample_jobs.json")
    
    try:
        students = [StudentFeatures(**s) for s in students_data]
        jobs = [JobFeatures(**j) for j in jobs_data]
    except Exception as e:
        logger.critical(f"Data validation failed against schema: {e}", exc_info=True)
        sys.exit(1)
        
    logger.info(f"Loaded {len(students)} students and {len(jobs)} jobs.")
    
    # 2. Compute Match Vectors
    all_vectors = []
    for student in students:
        for job in jobs:
            try:
                vector = compute_match_vector(student, job)
                all_vectors.append(vector.model_dump())
            except Exception as e:
                logger.error(f"Failed on Student {student.student_id} and Job {job.job_id}")
                
    if not all_vectors:
        logger.critical("No match vectors were computed. Exiting.")
        sys.exit(1)
        
    # 3. Save Metrics to JSON
    metrics_path = "logs/task02_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(all_vectors, f, indent=2)
    logger.info(f"Saved {len(all_vectors)} match vectors to {metrics_path}")
    
    # 4. Generate Visualizations (Matplotlib)
    logger.info("Generating visualization for matching features...")
    try:
        df = pd.DataFrame(all_vectors)
        
        # We will create a pivot table for Skill Overlap Ratio
        pivot_skill = df.pivot(index="student_id", columns="job_id", values="skill_overlap_ratio")
        
        # We will create a pivot table for Coding Threshold Met
        pivot_coding = df.pivot(index="student_id", columns="job_id", values="coding_threshold_met").astype(int)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        sns.heatmap(pivot_skill, annot=True, cmap="YlGnBu", ax=axes[0], vmin=0, vmax=1)
        axes[0].set_title("Skill Overlap Ratio")
        axes[0].set_xlabel("Job ID")
        axes[0].set_ylabel("Student ID")
        
        sns.heatmap(pivot_coding, annot=True, cmap="Blues", ax=axes[1], cbar=False)
        axes[1].set_title("Coding Threshold Met (1=Yes, 0=No)")
        axes[1].set_xlabel("Job ID")
        axes[1].set_ylabel("Student ID")
        
        plt.tight_layout()
        viz_path = "task02_matching_visualization.png"
        plt.savefig(viz_path, dpi=300)
        plt.close()
        
        logger.info(f"Saved visualization to {viz_path}")
    except Exception as e:
        logger.error(f"Failed to generate visualization: {e}", exc_info=True)
        # Not fatal, but we want to log it
        
    logger.info("Task 02 Pipeline completed successfully.")

if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception as e:
        logger.critical(f"Unhandled fatal error: {e}", exc_info=True)
        sys.exit(1)
