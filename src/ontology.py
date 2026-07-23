"""
ontology.py — Task 14: Skill Ontology for PlaceMux

Defines a two-level skill ontology:
  Level 1 — Domain  (e.g., "machine_learning", "cloud_devops")
  Level 2 — Cluster (e.g., "deep_learning_frameworks")

Provides:
  - SKILL_ONTOLOGY : nested dict — the full ontology definition
  - ALIAS_MAP      : dict mapping alias/variant -> canonical skill name
  - lookup()       : normalise a raw skill string to its canonical form
  - classify()     : map a canonical skill to its (domain, cluster)
  - feed_skills()  : map a list of raw skills into ontology-mapped records
  - summarise()    : aggregate ontology stats for a parsed profile

Standing instructions: robust error handling, structured logging,
NumPy-style docstrings, random_state=42.
"""

import os
import logging
from typing import List, Dict, Tuple, Optional

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


# ---------------------------------------------------------------------------
# Ontology Definition
# Domain -> Cluster -> [canonical skills]
# ---------------------------------------------------------------------------
SKILL_ONTOLOGY: Dict[str, Dict[str, List[str]]] = {
    "programming_languages": {
        "general_purpose": [
            "python", "java", "c++", "c#", "go", "rust", "ruby", "php", "scala",
        ],
        "scripting": [
            "javascript", "typescript", "bash", "r",
        ],
        "mobile": [
            "kotlin", "swift",
        ],
    },
    "web_development": {
        "frontend_frameworks": [
            "react", "angular", "vue",
        ],
        "backend_frameworks": [
            "node.js", "fastapi", "django", "flask", "spring boot",
        ],
        "mobile_frameworks": [
            "react native", "jetpack compose", "android",
        ],
        "api_patterns": [
            "rest api", "graphql", "microservices",
        ],
    },
    "data_and_ml": {
        "data_processing": [
            "pandas", "numpy", "spark", "hadoop",
        ],
        "machine_learning": [
            "scikit-learn", "lightgbm", "xgboost",
        ],
        "deep_learning_frameworks": [
            "tensorflow", "pytorch", "keras",
        ],
    },
    "databases": {
        "relational": [
            "sql", "postgresql", "mysql",
        ],
        "nosql": [
            "mongodb", "redis", "elasticsearch",
        ],
    },
    "cloud_and_devops": {
        "cloud_platforms": [
            "aws", "gcp", "azure",
        ],
        "containers_and_orchestration": [
            "docker", "kubernetes",
        ],
        "infrastructure_as_code": [
            "terraform",
        ],
        "ci_cd_and_vcs": [
            "git", "ci/cd", "jenkins", "github actions",
        ],
    },
    "data_visualization": {
        "bi_tools": [
            "tableau", "power bi", "excel",
        ],
    },
    "operating_systems": {
        "unix_like": [
            "linux",
        ],
    },
    "soft_skills": {
        "interpersonal": [
            "communication", "teamwork", "collaboration",
            "leadership", "negotiation", "mentoring", "conflict resolution",
        ],
        "cognitive": [
            "problem solving", "critical thinking", "decision making",
            "creativity", "attention to detail",
        ],
        "self_management": [
            "time management", "adaptability", "presentation",
        ],
    },
    "process_and_methodology": {
        "agile_methods": [
            "agile", "scrum",
        ],
    },
}


# ---------------------------------------------------------------------------
# Alias Map  (raw text variants -> canonical ontology skill name)
# ---------------------------------------------------------------------------
ALIAS_MAP: Dict[str, str] = {
    # Python variants
    "python3":          "python",
    "py":               "python",
    # JavaScript variants
    "js":               "javascript",
    "es6":              "javascript",
    "node":             "node.js",
    "nodejs":           "node.js",
    # React variants
    "reactjs":          "react",
    "react.js":         "react",
    # Database variants
    "postgres":         "postgresql",
    "mongo":            "mongodb",
    # Cloud variants
    "amazon web services": "aws",
    "google cloud":     "gcp",
    "gcp cloud":        "gcp",
    "microsoft azure":  "azure",
    # DevOps
    "github":           "git",
    "gitlab":           "git",
    "k8s":              "kubernetes",
    # ML
    "sklearn":          "scikit-learn",
    "lgbm":             "lightgbm",
    "xgb":              "xgboost",
    "tf":               "tensorflow",
    # Soft skill variants
    "team work":        "teamwork",
    "communication skills": "communication",
    "problem-solving":  "problem solving",
    "time-management":  "time management",
    "critical-thinking": "critical thinking",
}


# ---------------------------------------------------------------------------
# Internal lookup table (built once from SKILL_ONTOLOGY)
# canonical_skill -> (domain, cluster)
# ---------------------------------------------------------------------------
_SKILL_INDEX: Dict[str, Tuple[str, str]] = {}

def _build_index() -> None:
    """
    Populate _SKILL_INDEX from SKILL_ONTOLOGY.

    Called once at module load. Maps every canonical skill string to its
    (domain, cluster) tuple for O(1) lookup during inference.
    """
    for domain, clusters in SKILL_ONTOLOGY.items():
        for cluster, skills in clusters.items():
            for skill in skills:
                _SKILL_INDEX[skill.lower()] = (domain, cluster)

_build_index()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def lookup(raw_skill: str) -> str:
    """
    Normalise a raw skill string to its canonical ontology name.

    Applies alias resolution first, then lowercases and strips whitespace.
    Returns the canonical name, or the lowercased input if no alias exists.

    Parameters
    ----------
    raw_skill : str
        Raw skill string as extracted by the parser.

    Returns
    -------
    str
        Canonical skill name in lowercase.
    """
    if not raw_skill:
        return ""
    normalised = raw_skill.lower().strip()
    return ALIAS_MAP.get(normalised, normalised)


def classify(canonical_skill: str) -> Optional[Tuple[str, str]]:
    """
    Map a canonical skill to its (domain, cluster) in the ontology.

    Parameters
    ----------
    canonical_skill : str
        A canonical skill name (output of lookup()).

    Returns
    -------
    Tuple[str, str] or None
        (domain, cluster) if found; None if the skill is not in the ontology.
    """
    if canonical_skill is None:
        raise ValueError("classify() received None skill.")
    return _SKILL_INDEX.get(canonical_skill.lower().strip())


def feed_skills(
    raw_skills: List[str],
    source_id: str = "unknown",
    source_type: str = "resume",
) -> List[Dict]:
    """
    Map a list of raw parsed skills into ontology-enriched records.

    For each skill:
      1. Resolve alias -> canonical name
      2. Look up (domain, cluster) in ontology
      3. Mark skills not found in ontology as domain='unknown'

    This is the primary integration point between parser.py output and
    the ontology layer.

    Parameters
    ----------
    raw_skills : List[str]
        Skills as returned by parse_resume() or parse_job_description().
    source_id : str
        Student or job ID, used for traceability in logs.
    source_type : str
        One of 'resume' or 'job'. Used for logging context.

    Returns
    -------
    List[Dict]
        List of dicts, each with keys:
        raw, canonical, domain, cluster, in_ontology
    """
    if not raw_skills:
        logger.warning(f"[{source_id}] Empty skill list passed to feed_skills().")
        return []

    records = []
    unmapped = []

    for raw in raw_skills:
        try:
            canonical = lookup(raw)
            location  = classify(canonical)

            if location:
                domain, cluster = location
                in_ontology = True
            else:
                domain, cluster = "unknown", "unknown"
                in_ontology = False
                unmapped.append(raw)

            records.append({
                "raw":         raw,
                "canonical":   canonical,
                "domain":      domain,
                "cluster":     cluster,
                "in_ontology": in_ontology,
            })
        except Exception as e:
            logger.error(f"[{source_id}] Failed to process skill '{raw}': {e}")
            records.append({
                "raw":         raw,
                "canonical":   raw,
                "domain":      "error",
                "cluster":     "error",
                "in_ontology": False,
            })

    coverage = (len(records) - len(unmapped)) / len(records) * 100 if records else 0.0
    logger.info(
        f"[{source_id}] feed_skills({source_type}): "
        f"{len(records)} skills | "
        f"{len(records) - len(unmapped)} mapped | "
        f"{len(unmapped)} unmapped | "
        f"ontology coverage: {coverage:.1f}%"
    )
    if unmapped:
        logger.debug(f"[{source_id}] Unmapped skills: {unmapped}")

    return records


def summarise(ontology_records: List[Dict], source_id: str = "unknown") -> Dict:
    """
    Aggregate ontology-mapped skill records into a summary profile.

    Produces domain distribution, cluster breakdown, coverage rate,
    and lists of mapped and unmapped skills.

    Parameters
    ----------
    ontology_records : List[Dict]
        Output of feed_skills().
    source_id : str
        ID for logging.

    Returns
    -------
    Dict
        Summary with keys: total_skills, mapped, unmapped, coverage_pct,
        domain_distribution, cluster_breakdown, unmapped_skills.
    """
    if not ontology_records:
        logger.warning(f"[{source_id}] summarise() called with empty records.")
        return {}

    total   = len(ontology_records)
    mapped  = [r for r in ontology_records if r["in_ontology"]]
    unmapped_list = [r["raw"] for r in ontology_records if not r["in_ontology"]]

    domain_dist: Dict[str, int] = {}
    cluster_breakdown: Dict[str, Dict[str, int]] = {}

    for r in mapped:
        d, c = r["domain"], r["cluster"]
        domain_dist[d] = domain_dist.get(d, 0) + 1
        if d not in cluster_breakdown:
            cluster_breakdown[d] = {}
        cluster_breakdown[d][c] = cluster_breakdown[d].get(c, 0) + 1

    summary = {
        "source_id":           source_id,
        "total_skills":        total,
        "mapped_count":        len(mapped),
        "unmapped_count":      len(unmapped_list),
        "coverage_pct":        round(len(mapped) / total * 100, 1) if total > 0 else 0.0,
        "domain_distribution": domain_dist,
        "cluster_breakdown":   cluster_breakdown,
        "unmapped_skills":     unmapped_list,
    }

    logger.info(
        f"[{source_id}] Ontology summary: "
        f"{len(mapped)}/{total} mapped ({summary['coverage_pct']}%) | "
        f"domains: {list(domain_dist.keys())}"
    )
    return summary
