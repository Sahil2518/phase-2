import logging

# Ensure logs directory exists when running
import os
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
    Computes derived matching features from a Student and Job.
    """
    try:
        # 1. Skill Overlap Ratio (Jaccard similarity approximation using sets)
        student_skills = set(s.lower() for s in student.skills_hard)
        job_skills = set(s.lower() for s in job.required_skills)
        
        if len(job_skills) == 0:
            skill_overlap_ratio = 1.0 # No requirements means 100% overlap
        else:
            overlap = student_skills.intersection(job_skills)
            skill_overlap_ratio = len(overlap) / len(job_skills)
            
        # 2. Experience Gap
        experience_gap = student.years_experience - job.min_experience
        
        # 3. Salary Match Ratio
        # Higher is better, max salary job offers / expected salary
        if student.expected_salary > 0:
            salary_match_ratio = job.salary_max / student.expected_salary
        else:
            salary_match_ratio = 1.0
            
        # 4. Location Match
        location_match = (
            job.work_model.lower() == "remote" or 
            student.preferred_location.lower() == job.job_location.lower()
        )
        
        # 5. Education Met
        education_met = student.education_level >= job.min_education
        
        # 6. Competency Thresholds
        coding_threshold_met = student.coding_score >= job.min_coding_score
        communication_threshold_met = student.communication_score >= job.min_communication_score
        
        return MatchVector(
            student_id=student.student_id,
            job_id=job.job_id,
            skill_overlap_ratio=skill_overlap_ratio,
            experience_gap=experience_gap,
            salary_match_ratio=salary_match_ratio,
            location_match=location_match,
            education_met=education_met,
            coding_threshold_met=coding_threshold_met,
            communication_threshold_met=communication_threshold_met
        )
    except Exception as e:
        logger.error(f"Error computing match vector for Student {student.student_id} and Job {job.job_id}: {e}", exc_info=True)
        raise
