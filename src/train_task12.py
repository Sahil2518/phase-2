"""
train_task12.py — Task 12: Resume / JD Parsing v0 Demo Pipeline

Runs 3 sample resumes and 3 sample JDs through the parser, validates
that structured skills are produced, and saves all results to
logs/task12_parsed_output.json.

Standing instructions: robust error handling, structured logging.
"""

import os
import sys
import json
import logging
from datetime import datetime

# ---------------------------------------------------------------------------
# Logging Setup
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

OUTPUT_PATH = "logs/task12_parsed_output.json"

# ---------------------------------------------------------------------------
# Sample Data
# ---------------------------------------------------------------------------

SAMPLE_RESUMES = [
    {
        "id": "STU-001",
        "text": """
        Alice Sharma | Machine Learning Engineer
        4 years of experience in data science and ML engineering.
        Education: Master of Science (M.Sc) in Artificial Intelligence.
        Technical Skills: Python, TensorFlow, PyTorch, Scikit-learn, SQL,
        Pandas, NumPy, Docker, Git, AWS.
        Soft Skills: Strong communication, collaboration, and problem solving abilities.
        Led a team of 3 engineers on a production ML pipeline project.
        """,
    },
    {
        "id": "STU-002",
        "text": """
        Bob Patel | Full Stack Developer
        2 years of experience building web applications.
        Education: Bachelor of Engineering (B.E.) in Computer Science.
        Skills: JavaScript, TypeScript, React, Node.js, MongoDB, REST API, Git, Docker.
        Soft Skills: Teamwork, attention to detail, time management.
        """,
    },
    {
        "id": "STU-003",
        "text": """
        Carol Kim | Android Developer
        3+ years of experience in mobile application development.
        Education: Bachelor of Technology (B.Tech) in Information Technology.
        Technical Skills: Kotlin, Java, Android, Jetpack Compose, REST API, Git, Firebase.
        Soft Skills: Adaptability, creativity, presentation skills.
        """,
    },
]

SAMPLE_JDS = [
    {
        "id": "JOB-001",
        "text": """
        Senior ML Engineer
        We are looking for a Senior Machine Learning Engineer to join our AI team.
        Required:
        5+ years of experience. Master's degree or higher in CS, AI, or related field.
        Must have: Python, TensorFlow, PyTorch, SQL, Docker, Git, AWS.
        Salary: $120,000 - $160,000. Hybrid work model.
        Preferred:
        Nice to have: Kubernetes, Spark, GraphQL.
        """,
    },
    {
        "id": "JOB-002",
        "text": """
        Full Stack Engineer
        We need a talented full stack developer to build scalable web products.
        Required:
        2+ years of experience. Bachelor's degree required.
        Must have: JavaScript, React, Node.js, REST API, MongoDB, Git.
        Salary: $80,000 - $110,000. Remote position.
        Preferred:
        Nice to have: TypeScript, Docker, AWS.
        """,
    },
    {
        "id": "JOB-003",
        "text": """
        Android Engineer
        Join our mobile team and build cutting-edge Android apps for millions of users.
        Required:
        3 years of experience minimum. Bachelor's degree in Computer Science or related.
        Must have: Kotlin, Android, Jetpack Compose, REST API, Git.
        Salary: $90,000 - $120,000. On-site role.
        Preferred:
        Nice to have: Java, Firebase, CI/CD.
        """,
    },
]


# ---------------------------------------------------------------------------
# Pipeline Steps
# ---------------------------------------------------------------------------

def run_resume_parsing(parser_module) -> list:
    """
    Parse all sample resumes and return a list of structured results.

    Parameters
    ----------
    parser_module : module
        The imported parser module.

    Returns
    -------
    list
        List of parsed StudentFeatures dicts.
    """
    results = []
    logger.info("=" * 50)
    logger.info("PARSING RESUMES")
    logger.info("=" * 50)
    for sample in SAMPLE_RESUMES:
        try:
            parsed = parser_module.parse_resume(sample["text"], sample["id"])
            assert len(parsed["skills_hard"]) > 0, f"No hard skills found for {sample['id']}"
            results.append(parsed)
            logger.info(f"  [{sample['id']}] hard_skills={parsed['skills_hard']}")
        except Exception as e:
            logger.error(f"  [{sample['id']}] Parsing failed: {e}", exc_info=True)
    return results


def run_jd_parsing(parser_module) -> list:
    """
    Parse all sample JDs and return a list of structured results.

    Parameters
    ----------
    parser_module : module
        The imported parser module.

    Returns
    -------
    list
        List of parsed JobFeatures dicts.
    """
    results = []
    logger.info("=" * 50)
    logger.info("PARSING JOB DESCRIPTIONS")
    logger.info("=" * 50)
    for sample in SAMPLE_JDS:
        try:
            parsed = parser_module.parse_job_description(sample["text"], sample["id"])
            assert len(parsed["required_skills"]) > 0, f"No required skills found for {sample['id']}"
            results.append(parsed)
            logger.info(f"  [{sample['id']}] required_skills={parsed['required_skills']}")
            logger.info(f"  [{sample['id']}] preferred_skills={parsed['preferred_skills']}")
        except Exception as e:
            logger.error(f"  [{sample['id']}] Parsing failed: {e}", exc_info=True)
    return results


def save_results(resumes: list, jds: list) -> None:
    """
    Save all parsed results to a JSON output file.

    Parameters
    ----------
    resumes : list
        Parsed StudentFeatures dicts.
    jds : list
        Parsed JobFeatures dicts.
    """
    output = {
        "task": "Task 12 — Resume/JD Parsing v0",
        "timestamp": datetime.now().isoformat(),
        "parsed_resumes": resumes,
        "parsed_job_descriptions": jds,
        "summary": {
            "resumes_parsed": len(resumes),
            "jds_parsed": len(jds),
            "total_parsed": len(resumes) + len(jds),
        },
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Results saved to {OUTPUT_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_pipeline() -> None:
    """
    End-to-end parsing v0 demo pipeline:
    1. Import parser module
    2. Parse 3 sample resumes -> structured StudentFeatures
    3. Parse 3 sample JDs -> structured JobFeatures
    4. Validate output (assert skills list non-empty)
    5. Save all results to logs/task12_parsed_output.json
    """
    logger.info("=" * 60)
    logger.info("  PlaceMux Task 12 — Resume/JD Parsing v0")
    logger.info("  Goal: Produce structured skills from raw text")
    logger.info("=" * 60)

    # Guard: import parser
    try:
        import parser as _parser_module  # noqa — avoid shadowing builtin
    except ImportError:
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
            from src import parser as _parser_module
        except ImportError as e:
            logger.critical(f"Cannot import parser module: {e}")
            sys.exit(1)

    parsed_resumes = run_resume_parsing(_parser_module)
    parsed_jds     = run_jd_parsing(_parser_module)

    # Final validation
    assert len(parsed_resumes) == 3, f"Expected 3 parsed resumes, got {len(parsed_resumes)}"
    assert len(parsed_jds)     == 3, f"Expected 3 parsed JDs, got {len(parsed_jds)}"
    logger.info("All assertions passed. Parsing v0 is demoable.")

    save_results(parsed_resumes, parsed_jds)
    logger.info("Task 12 pipeline complete.")


def main():
    try:
        run_pipeline()
    except FileNotFoundError as e:
        logger.critical(f"Missing required file: {e}")
        sys.exit(1)
    except AssertionError as e:
        logger.critical(f"Validation failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Unhandled fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
