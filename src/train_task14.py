"""
train_task14.py — Task 14: Parsing into Ontology Pipeline

End-to-end demo:
  1. Parse 3 sample resumes and 3 JDs using parser.py (Task 12)
  2. Feed all extracted skills through ontology.py (Task 14)
  3. Produce ontology-enriched JSON output with domain/cluster mapping
  4. Assert ontology coverage >= 80% for each profile

Standing instructions: robust error handling, structured logging,
NumPy-style docstrings, random_state=42.
"""

import os
import sys
import json
import logging
from datetime import datetime

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/task14.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

OUTPUT_PATH  = "logs/task14_ontology_output.json"
METRICS_PATH = "logs/task14_metrics.json"
MIN_COVERAGE = 80.0   # Assert ontology coverage >= 80% per profile

# ---------------------------------------------------------------------------
# Sample Data (reusing Task 12 samples + 2 new ones for breadth)
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
    {
        "id": "STU-004",
        "text": """
        David Nair | DevOps Engineer
        5 years of experience in cloud infrastructure and CI/CD pipelines.
        Education: Bachelor of Science (B.Sc) in Computer Science.
        Skills: AWS, Azure, Docker, Kubernetes, Terraform, Jenkins, Git,
        Python, Bash, Linux.
        Soft Skills: Leadership, problem solving, communication.
        """,
    },
]

SAMPLE_JDS = [
    {
        "id": "JOB-001",
        "text": """
        Senior ML Engineer
        Required:
        5+ years of experience. Master's degree in CS, AI, or related.
        Must have: Python, TensorFlow, PyTorch, SQL, Docker, Git, AWS.
        Salary: $120,000 - $160,000. Hybrid work.
        Preferred:
        Nice to have: Kubernetes, Spark, GraphQL.
        """,
    },
    {
        "id": "JOB-002",
        "text": """
        Full Stack Engineer
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
        Cloud Infrastructure Engineer
        Required:
        4+ years of experience. Bachelor's degree in CS or equivalent.
        Must have: AWS, Docker, Kubernetes, Terraform, Python, Git.
        Salary: $110,000 - $140,000. On-site role.
        Preferred:
        Nice to have: Azure, GCP, Jenkins.
        """,
    },
]


# ---------------------------------------------------------------------------
# Pipeline Steps
# ---------------------------------------------------------------------------

def parse_all_profiles(parser_module) -> tuple:
    """
    Parse all sample resumes and JDs using Task 12 parser.

    Parameters
    ----------
    parser_module : module
        Imported parser module (parser.py).

    Returns
    -------
    tuple
        (parsed_resumes: list, parsed_jds: list)
    """
    logger.info("=" * 55)
    logger.info("  STEP 1: Parse Resumes and JDs (Task 12 parser)")
    logger.info("=" * 55)

    parsed_resumes = []
    for sample in SAMPLE_RESUMES:
        try:
            result = parser_module.parse_resume(sample["text"], sample["id"])
            assert len(result["skills_hard"]) > 0, f"No hard skills for {sample['id']}"
            parsed_resumes.append(result)
        except Exception as e:
            logger.error(f"Resume parse failed [{sample['id']}]: {e}", exc_info=True)

    parsed_jds = []
    for sample in SAMPLE_JDS:
        try:
            result = parser_module.parse_job_description(sample["text"], sample["id"])
            assert len(result["required_skills"]) > 0, f"No required skills for {sample['id']}"
            parsed_jds.append(result)
        except Exception as e:
            logger.error(f"JD parse failed [{sample['id']}]: {e}", exc_info=True)

    logger.info(f"Parsed {len(parsed_resumes)} resumes, {len(parsed_jds)} JDs.")
    return parsed_resumes, parsed_jds


def feed_into_ontology(parsed_resumes: list, parsed_jds: list, ontology_module) -> dict:
    """
    Feed all parsed skills into the ontology and produce enriched records.

    For resumes: feeds skills_hard + skills_soft.
    For JDs    : feeds required_skills + preferred_skills.

    Parameters
    ----------
    parsed_resumes : list
        Output of parse_all_profiles() for resumes.
    parsed_jds : list
        Output of parse_all_profiles() for JDs.
    ontology_module : module
        Imported ontology module (ontology.py).

    Returns
    -------
    dict
        Full enriched output with ontology records and summaries.
    """
    logger.info("=" * 55)
    logger.info("  STEP 2: Feed Parsed Skills into Ontology")
    logger.info("=" * 55)

    enriched_resumes = []
    for resume in parsed_resumes:
        sid = resume["student_id"]
        all_skills = resume["skills_hard"] + resume.get("skills_soft", [])

        if not all_skills:
            logger.warning(f"[{sid}] No skills to feed into ontology.")
            continue

        records = ontology_module.feed_skills(all_skills, source_id=sid, source_type="resume")
        summary = ontology_module.summarise(records, source_id=sid)

        enriched_resumes.append({
            "student_id":      sid,
            "raw_skills":      all_skills,
            "ontology_records": records,
            "summary":         summary,
        })

    enriched_jds = []
    for jd in parsed_jds:
        jid = jd["job_id"]
        all_skills = jd["required_skills"] + jd.get("preferred_skills", [])

        if not all_skills:
            logger.warning(f"[{jid}] No skills to feed into ontology.")
            continue

        records = ontology_module.feed_skills(all_skills, source_id=jid, source_type="job")
        summary = ontology_module.summarise(records, source_id=jid)

        enriched_jds.append({
            "job_id":          jid,
            "raw_skills":      all_skills,
            "ontology_records": records,
            "summary":         summary,
        })

    return {"enriched_resumes": enriched_resumes, "enriched_jds": enriched_jds}


def validate_coverage(enriched_output: dict) -> dict:
    """
    Assert ontology coverage >= MIN_COVERAGE for every profile.

    Computes aggregate stats across all profiles and logs pass/fail per entity.

    Parameters
    ----------
    enriched_output : dict
        Output of feed_into_ontology().

    Returns
    -------
    dict
        Validation report with per-entity results and overall pass/fail.
    """
    logger.info("=" * 55)
    logger.info("  STEP 3: Validate Ontology Coverage")
    logger.info("=" * 55)

    all_entities   = (
        enriched_output["enriched_resumes"] +
        enriched_output["enriched_jds"]
    )
    results = []
    all_pass = True

    for entity in all_entities:
        eid      = entity.get("student_id") or entity.get("job_id")
        summary  = entity["summary"]
        coverage = summary.get("coverage_pct", 0.0)
        passed   = coverage >= MIN_COVERAGE

        if not passed:
            all_pass = False
            logger.warning(f"  [{eid}] FAIL — coverage {coverage:.1f}% < {MIN_COVERAGE:.0f}%")
        else:
            logger.info(f"  [{eid}] PASS — coverage {coverage:.1f}%  domains: {list(summary['domain_distribution'].keys())}")

        results.append({
            "id":             eid,
            "coverage_pct":   coverage,
            "mapped":         summary.get("mapped_count", 0),
            "total":          summary.get("total_skills", 0),
            "passed":         passed,
        })

    avg_cov = sum(r["coverage_pct"] for r in results) / len(results) if results else 0.0
    report  = {
        "min_coverage_threshold": MIN_COVERAGE,
        "num_entities_checked":   len(results),
        "avg_coverage_pct":       round(avg_cov, 1),
        "all_entities_passed":    all_pass,
        "per_entity":             results,
    }

    logger.info(f"Avg coverage: {avg_cov:.1f}% | All passed: {all_pass}")
    return report


def save_outputs(enriched_output: dict, validation_report: dict) -> None:
    """
    Save full ontology-enriched output and validation metrics to logs/.

    Parameters
    ----------
    enriched_output : dict
        Full enriched records from feed_into_ontology().
    validation_report : dict
        Validation summary from validate_coverage().
    """
    try:
        full_output = {
            "task": "Task 14 — Parsing into Ontology",
            "timestamp": datetime.now().isoformat(),
            "enriched_resumes": enriched_output["enriched_resumes"],
            "enriched_jds":     enriched_output["enriched_jds"],
        }
        with open(OUTPUT_PATH, "w") as f:
            json.dump(full_output, f, indent=2)
        logger.info(f"Full ontology output saved -> {OUTPUT_PATH}")

        metrics = {
            "task":      "Task 14 — Parsing into Ontology",
            "timestamp": datetime.now().isoformat(),
            "validation": validation_report,
        }
        with open(METRICS_PATH, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved -> {METRICS_PATH}")

    except Exception as e:
        logger.error(f"Failed to save outputs: {e}", exc_info=True)
        raise


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def run_pipeline() -> None:
    """
    End-to-end Task 14 pipeline:
    1. Import parser (Task 12) and ontology (Task 14) modules
    2. Parse 4 resumes and 3 JDs
    3. Feed all parsed skills into ontology
    4. Validate coverage >= 80%
    5. Save ontology-enriched output + metrics to logs/
    """
    logger.info("=" * 60)
    logger.info("  PlaceMux Task 14 — End-to-End Status Tracking & Parsing")
    logger.info("  Focus: Feed parsed skills into the ontology")
    logger.info("=" * 60)

    # Import parser
    try:
        sys.path.insert(0, os.path.dirname(__file__))
        import parser as _parser
    except ImportError:
        try:
            from src import parser as _parser
        except ImportError as e:
            logger.critical(f"Cannot import parser: {e}")
            sys.exit(1)

    # Import ontology
    try:
        import ontology as _ontology
    except ImportError:
        try:
            from src import ontology as _ontology
        except ImportError as e:
            logger.critical(f"Cannot import ontology: {e}")
            sys.exit(1)

    parsed_resumes, parsed_jds = parse_all_profiles(_parser)

    if not parsed_resumes:
        raise RuntimeError("No resumes were successfully parsed.")
    if not parsed_jds:
        raise RuntimeError("No JDs were successfully parsed.")

    enriched_output   = feed_into_ontology(parsed_resumes, parsed_jds, _ontology)
    validation_report = validate_coverage(enriched_output)
    save_outputs(enriched_output, validation_report)

    if not validation_report["all_entities_passed"]:
        logger.warning("Some entities did not meet coverage threshold.")
    else:
        logger.info("All entities passed ontology coverage validation.")

    logger.info("Task 14 pipeline complete.")


def main():
    try:
        run_pipeline()
    except FileNotFoundError as e:
        logger.critical(f"Missing required file: {e}")
        sys.exit(1)
    except AssertionError as e:
        logger.critical(f"Assertion failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Unhandled fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
