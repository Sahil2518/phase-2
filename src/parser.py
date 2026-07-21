"""
parser.py — Task 12: Resume / Job Description Parsing v0

Rule-based parser that extracts structured skills and features from raw
resume text and job description text. This is the v0 foundation layer;
it uses regex patterns and a curated keyword vocabulary.

Output types: StudentFeatures, JobFeatures (from model_schemas.py)

Standing instructions: robust error handling, structured logging,
NumPy-style docstrings.
"""

import os
import re
import uuid
import logging
from typing import List, Tuple, Optional

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/task12.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Skill Vocabulary
# ---------------------------------------------------------------------------

HARD_SKILLS_VOCAB: List[str] = [
    # Languages
    "python", "java", "kotlin", "swift", "javascript", "typescript",
    "c++", "c#", "go", "rust", "r", "scala", "ruby", "php",
    # Web / Mobile
    "react", "react native", "angular", "vue", "node.js", "fastapi",
    "django", "flask", "spring boot", "android", "jetpack compose",
    # Data / ML
    "sql", "postgresql", "mysql", "mongodb", "redis", "elasticsearch",
    "pandas", "numpy", "scikit-learn", "lightgbm", "xgboost", "tensorflow",
    "pytorch", "keras", "spark", "hadoop",
    # Cloud / DevOps
    "aws", "gcp", "azure", "docker", "kubernetes", "terraform", "git",
    "ci/cd", "jenkins", "github actions",
    # Other
    "rest api", "graphql", "microservices", "agile", "scrum",
    "linux", "bash", "excel", "tableau", "power bi",
]

SOFT_SKILLS_VOCAB: List[str] = [
    "communication", "leadership", "teamwork", "problem solving",
    "critical thinking", "time management", "adaptability", "creativity",
    "attention to detail", "collaboration", "presentation", "negotiation",
    "mentoring", "conflict resolution", "decision making",
]

EDUCATION_KEYWORDS = {
    4: ["phd", "ph.d", "doctorate", "doctoral"],
    3: ["master", "msc", "m.sc", "mba", "m.tech", "m.e."],
    2: ["bachelor", "bsc", "b.sc", "b.tech", "b.e.", "undergraduate", "degree"],
    1: ["high school", "diploma", "12th", "hsc"],
}

# ---------------------------------------------------------------------------
# Shared Helpers
# ---------------------------------------------------------------------------

def _extract_skills(text: str, vocab: List[str]) -> List[str]:
    """
    Extract skills from text by matching against a vocabulary list.

    Uses case-insensitive whole-phrase matching to avoid partial matches.

    Parameters
    ----------
    text : str
        Raw input text (resume or JD).
    vocab : List[str]
        List of skill phrases to search for.

    Returns
    -------
    List[str]
        Deduplicated list of matched skills in lowercase.
    """
    text_lower = text.lower()
    found = []
    for skill in vocab:
        # Use word-boundary-aware pattern for single words; phrase match for multi-word
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text_lower):
            if skill not in found:
                found.append(skill)
    return found


def _extract_years_experience(text: str) -> float:
    """
    Infer total years of experience from text patterns.

    Looks for patterns like '3 years', '5+ years', '2-4 years experience'.

    Parameters
    ----------
    text : str
        Raw text to search.

    Returns
    -------
    float
        Estimated years of experience. Returns 0.0 if not found.
    """
    patterns = [
        r"(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s+)?(?:experience|exp)",
        r"(\d+)\s*-\s*(\d+)\s*(?:years?|yrs?)",
        r"(\d+)\s*(?:years?|yrs?)\s+(?:of\s+)?(?:work|professional|industry)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            # If range (e.g., 2-4), take the lower bound
            return float(match.group(1))
    return 0.0


def _extract_education_level(text: str) -> int:
    """
    Infer highest education level from text using keyword matching.

    Returns an ordinal encoding: 1=HS/Diploma, 2=BS, 3=MS, 4=PhD.

    Parameters
    ----------
    text : str
        Raw text to search.

    Returns
    -------
    int
        Ordinal education level (1-4). Returns 1 if not found.
    """
    text_lower = text.lower()
    for level in sorted(EDUCATION_KEYWORDS.keys(), reverse=True):
        for keyword in EDUCATION_KEYWORDS[level]:
            if keyword in text_lower:
                return level
    return 1


def _extract_salary_range(text: str) -> Tuple[float, float]:
    """
    Extract salary range from text patterns like '$80,000 - $120,000' or '80k-120k'.

    Parameters
    ----------
    text : str
        Raw text to search.

    Returns
    -------
    Tuple[float, float]
        (salary_min, salary_max). Returns (0.0, 0.0) if not found.
    """
    # Match patterns like $80,000 - $120,000 or 80k to 120k
    patterns = [
        r"\$(\d[\d,]*)\s*[-–to]+\s*\$(\d[\d,]*)",
        r"(\d+)k\s*[-–to]+\s*(\d+)k",
        r"(\d[\d,]*)\s*(?:lpa|per\s+annum|annually)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            s_min = float(match.group(1).replace(",", ""))
            if "k" in pattern:
                s_min *= 1000
            try:
                s_max = float(match.group(2).replace(",", ""))
                if "k" in pattern:
                    s_max *= 1000
            except IndexError:
                s_max = s_min * 1.3
            return s_min, s_max
    return 0.0, 0.0


def _extract_section(text: str, section_keywords: List[str]) -> str:
    """
    Extract the text belonging to a specific section in a JD.

    Looks for lines starting with section keywords and returns following content
    until the next section or end of text.

    Parameters
    ----------
    text : str
        Full JD text.
    section_keywords : List[str]
        Keywords that identify the section header (e.g., ['required', 'must have']).

    Returns
    -------
    str
        The extracted section text, or empty string if not found.
    """
    lines = text.split("\n")
    capture = False
    captured = []
    for line in lines:
        line_lower = line.lower().strip()
        if any(kw in line_lower for kw in section_keywords):
            capture = True
            continue
        if capture:
            # Stop at next section header (short line ending with colon)
            if line.strip().endswith(":") and len(line.strip()) < 50:
                break
            captured.append(line)
    return " ".join(captured)


# ---------------------------------------------------------------------------
# Core Parsers
# ---------------------------------------------------------------------------

def parse_resume(text: str, student_id: Optional[str] = None) -> dict:
    """
    Parse a raw resume text string into a structured StudentFeatures dict.

    Extracts: hard skills, soft skills, years of experience, education level.

    Parameters
    ----------
    text : str
        Raw resume text (plain text, not PDF bytes).
    student_id : str, optional
        Identifier for this student. Auto-generated UUID if not provided.

    Returns
    -------
    dict
        Structured dictionary compatible with StudentFeatures schema.

    Raises
    ------
    ValueError
        If text is empty or None.
    """
    if not text or not text.strip():
        raise ValueError("Resume text cannot be empty.")

    sid = student_id or f"STU-{uuid.uuid4().hex[:8].upper()}"
    logger.info(f"Parsing resume for student_id={sid}...")

    try:
        hard_skills   = _extract_skills(text, HARD_SKILLS_VOCAB)
        soft_skills   = _extract_skills(text, SOFT_SKILLS_VOCAB)
        years_exp     = _extract_years_experience(text)
        edu_level     = _extract_education_level(text)

        result = {
            "student_id":         sid,
            "skills_hard":        hard_skills,
            "skills_soft":        soft_skills,
            "years_experience":   years_exp,
            "education_level":    edu_level,
            "expected_salary":    0.0,   # Not typically in resume; left for user to fill
            "preferred_location": "",
            "remote_preference":  "On-site",
            "coding_score":       0.0,
            "communication_score": 0.0,
        }

        logger.info(
            f"Resume parsed — student_id={sid} | "
            f"hard_skills={len(hard_skills)} | soft_skills={len(soft_skills)} | "
            f"years_exp={years_exp} | edu_level={edu_level}"
        )
        return result

    except Exception as e:
        logger.error(f"Resume parsing failed for student_id={sid}: {e}", exc_info=True)
        raise


def parse_job_description(text: str, job_id: Optional[str] = None) -> dict:
    """
    Parse a raw job description text string into a structured JobFeatures dict.

    Extracts: required skills, preferred skills, min experience, salary range,
    education level, and work model.

    Parameters
    ----------
    text : str
        Raw job description text (plain text).
    job_id : str, optional
        Identifier for this job. Auto-generated UUID if not provided.

    Returns
    -------
    dict
        Structured dictionary compatible with JobFeatures schema.

    Raises
    ------
    ValueError
        If text is empty or None.
    """
    if not text or not text.strip():
        raise ValueError("Job description text cannot be empty.")

    jid = job_id or f"JOB-{uuid.uuid4().hex[:8].upper()}"
    logger.info(f"Parsing job description for job_id={jid}...")

    try:
        # Try to extract skills from specific sections first; fall back to full text
        required_section  = _extract_section(text, ["required", "must have", "must-have", "responsibilities"])
        preferred_section = _extract_section(text, ["preferred", "nice to have", "nice-to-have", "bonus"])

        required_skills = _extract_skills(required_section, HARD_SKILLS_VOCAB) if required_section.strip() else []
        # Fall back to full text if section extraction found nothing
        if not required_skills:
            required_skills = _extract_skills(text, HARD_SKILLS_VOCAB)

        preferred_skills = _extract_skills(preferred_section, HARD_SKILLS_VOCAB) if preferred_section.strip() else []
        # Remove overlap: preferred should not repeat required
        preferred_skills = [s for s in preferred_skills if s not in required_skills]

        min_exp   = _extract_years_experience(text)
        edu_level = _extract_education_level(text)
        sal_min, sal_max = _extract_salary_range(text)

        # Detect work model
        text_lower = text.lower()
        if "remote" in text_lower:
            work_model = "Remote"
        elif "hybrid" in text_lower:
            work_model = "Hybrid"
        else:
            work_model = "On-site"

        result = {
            "job_id":                    jid,
            "required_skills":           required_skills,
            "preferred_skills":          preferred_skills,
            "min_experience":            min_exp,
            "max_experience":            None,
            "min_education":             edu_level,
            "salary_min":                sal_min,
            "salary_max":                sal_max,
            "job_location":              "",
            "work_model":                work_model,
            "min_coding_score":          0.0,
            "min_communication_score":   0.0,
        }

        logger.info(
            f"JD parsed — job_id={jid} | "
            f"required_skills={len(required_skills)} | preferred_skills={len(preferred_skills)} | "
            f"min_exp={min_exp} | salary=${sal_min:,.0f}-${sal_max:,.0f} | work_model={work_model}"
        )
        return result

    except Exception as e:
        logger.error(f"JD parsing failed for job_id={jid}: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    # Quick smoke test
    sample_resume = """
    John Doe | Software Engineer
    5 years of experience in backend development.
    Education: Bachelor of Technology (B.Tech) in Computer Science.
    Skills: Python, FastAPI, PostgreSQL, Docker, Git, AWS, REST API.
    Soft skills: Strong communication and teamwork. Leadership in cross-functional projects.
    """
    sample_jd = """
    Senior Backend Engineer
    Required:
    3+ years of experience. Bachelor's degree or higher.
    Must have: Python, Django, PostgreSQL, Docker, REST API.
    Salary: $90,000 - $130,000. Remote position.
    Preferred:
    Nice to have: Kubernetes, AWS, GraphQL.
    """
    print("\n--- Resume Parse ---")
    import json
    print(json.dumps(parse_resume(sample_resume, "STU-SMOKE"), indent=2))
    print("\n--- JD Parse ---")
    print(json.dumps(parse_job_description(sample_jd, "JOB-SMOKE"), indent=2))
