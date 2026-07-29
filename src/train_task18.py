"""
train_task18.py - Task 18: Admin Console & Review Queue
                  Strengthen recommendation explainability.

Pipeline:
  1. Load ranker_v2 model.
  2. Build synthetic cohort (4 students) and job pool (3 postings).
  3. Generate enriched explanations for every (student, job) pair using `rich_explainer.py`.
  4. Aggregate explanations and populate `src/templates/review_queue.html`.
  5. Save `logs/task18_explainability_report.json` and `logs/task18_metrics.json`.
  6. Save `logs/review_queue.html` (the admin console).

Standing instructions: robust error handling, structured logging,
NumPy-style docstrings, random_state=42.
"""

import os
import sys
import json
import logging
import datetime
from typing import List, Dict

# ---------------------------------------------------------------------------
# Logging & Setup
# ---------------------------------------------------------------------------
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/task18.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Constants
RANDOM_STATE    = 42
REPORT_PATH     = "logs/task18_explainability_report.json"
METRICS_PATH    = "logs/task18_metrics.json"
HTML_OUT_PATH   = "logs/review_queue.html"
TEMPLATE_PATH   = "src/templates/review_queue.html"

# ---------------------------------------------------------------------------
# Synthetic Data (4 Students x 3 Jobs)
# ---------------------------------------------------------------------------
STUDENTS = [
    {"student_id":"STU-T18-001","skills_hard":["python","tensorflow","sql","pandas","docker"],"skills_soft":["analytical thinking"],"years_experience":2.5,"education_level":3,"expected_salary":110000.0,"preferred_location":"Bangalore","remote_preference":"Hybrid","coding_score":0.85,"communication_score":0.75},
    {"student_id":"STU-T18-002","skills_hard":["javascript","react","node.js","git"],"skills_soft":["teamwork"],"years_experience":1.5,"education_level":2,"expected_salary":85000.0,"preferred_location":"Remote","remote_preference":"Remote","coding_score":0.70,"communication_score":0.80},
    {"student_id":"STU-T18-003","skills_hard":["aws","docker","kubernetes","terraform","python","linux"],"skills_soft":["leadership"],"years_experience":4.0,"education_level":2,"expected_salary":130000.0,"preferred_location":"Pune","remote_preference":"Hybrid","coding_score":0.80,"communication_score":0.85},
    {"student_id":"STU-T18-004","skills_hard":["kotlin","java","android","git"],"skills_soft":["adaptability"],"years_experience":2.0,"education_level":2,"expected_salary":80000.0,"preferred_location":"Mumbai","remote_preference":"On-site","coding_score":0.78,"communication_score":0.70},
]

JOBS = [
    {"job_id":"JOB-T18-001","required_skills":["python","tensorflow","sql","docker"],"preferred_skills":["aws"],"min_experience":2.0,"max_experience":None,"min_education":3,"salary_min":90000.0,"salary_max":130000.0,"job_location":"Bangalore","work_model":"Hybrid","min_coding_score":0.80,"min_communication_score":0.70},
    {"job_id":"JOB-T18-002","required_skills":["javascript","react","node.js","mongodb"],"preferred_skills":["typescript"],"min_experience":1.0,"max_experience":None,"min_education":2,"salary_min":75000.0,"salary_max":100000.0,"job_location":"Remote","work_model":"Remote","min_coding_score":0.65,"min_communication_score":0.75},
    {"job_id":"JOB-T18-003","required_skills":["aws","docker","kubernetes","terraform","python"],"preferred_skills":["azure"],"min_experience":3.5,"max_experience":None,"min_education":2,"salary_min":110000.0,"salary_max":150000.0,"job_location":"Pune","work_model":"Hybrid","min_coding_score":0.75,"min_communication_score":0.80},
]

# ---------------------------------------------------------------------------
# Pipeline Steps
# ---------------------------------------------------------------------------
def load_ranker_model():
    """Load best available ranker (v2 preferred, v1 fallback)."""
    from src.ranker import load_ranker
    pkls = [f for f in os.listdir("models") if f.startswith("ranker_") and f.endswith(".pkl")]
    if not pkls:
        raise FileNotFoundError("No ranker model found in models/.")
    pkls.sort(reverse=True)
    path = os.path.join("models", pkls[0])
    logger.info(f"Loading model: {path}")
    return load_ranker(path)

def build_profiles():
    """Instantiate Pydantic StudentFeatures and JobFeatures."""
    from src.model_schemas import StudentFeatures, JobFeatures
    students = [StudentFeatures(**s) for s in STUDENTS]
    jobs     = [JobFeatures(**j)     for j in JOBS]
    logger.info(f"Built {len(students)} student profiles and {len(jobs)} job profiles.")
    return students, jobs

def generate_enriched_explanations(model, students, jobs) -> List[Dict]:
    """
    Generate enriched explanations for all student-job pairs.
    """
    from src.match_vectors import compute_match_vector
    from src.ranker import score_pair
    from src.rich_explainer import enrich_recommendation

    # Compute missing skills for all jobs for cross-job upskilling hints
    all_jobs_missing = {}
    for job in jobs:
        for skill in job.required_skills:
            if skill.lower() not in all_jobs_missing:
                all_jobs_missing[skill.lower()] = []
            all_jobs_missing[skill.lower()].append(job.job_id)

    results = []
    cohort_size = len(students)

    for job in jobs:
        # First pass to compute scores and rank students for this job
        job_pairs = []
        for student in students:
            try:
                mv = compute_match_vector(student, job)
                score, feature_row = score_pair(model, mv)
                job_pairs.append((student, score, feature_row))
            except Exception as e:
                logger.error(f"Error scoring {student.student_id} against {job.job_id}: {e}")

        # Sort by score descending to assign ranks
        job_pairs.sort(key=lambda x: x[1], reverse=True)

        # Second pass to enrich
        for rank_idx, (student, score, feature_row) in enumerate(job_pairs):
            try:
                enriched = enrich_recommendation(
                    model=model,
                    student=student,
                    job=job,
                    score=score,
                    feature_row=feature_row,
                    rank_in_cohort=rank_idx + 1,
                    cohort_size=cohort_size,
                    all_jobs_missing=all_jobs_missing,
                )
                results.append(enriched)
            except Exception as e:
                logger.error(f"Error enriching {student.student_id} against {job.job_id}: {e}")

    logger.info(f"Generated {len(results)} enriched explanations.")
    return results

def render_html_dashboard(results: List[Dict]):
    """
    Populate the HTML template with enriched explanation cards.
    """
    if not os.path.exists(TEMPLATE_PATH):
        raise FileNotFoundError(f"Template not found: {TEMPLATE_PATH}")

    with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
        template = f.read()

    shortlisted_count = sum(1 for r in results if r.get("shortlist"))
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cards_html = ""
    for r in results:
        # Match Tier styling
        bg_color = r["match_tier_color"]
        tier_html = f"<div class='tier-badge' style='background: {bg_color}22; color: {bg_color}; border: 1px solid {bg_color}44;'>{r.get('match_tier_emoji','')} {r['match_tier']} ({r['score']:.2f})</div>"
        
        # Skills Grid
        present_html = "".join([f"<li class='skill-present'>✓ {s}</li>" for s in r['present_skills']])
        missing_html = "".join([f"<li class='skill-missing'>✗ {s}</li>" for s in r['missing_skills']])
        if not missing_html:
            missing_html = "<li style='color: #94a3b8;'>None missing</li>"

        # Feature Bars
        feature_bars_html = ""
        for f in r.get("feature_contributions", [])[:4]:  # Top 4 features
            val_pct = max(0, min(100, f['value'] * 100))
            color = "#4ade80" if f['verdict'] == "strength" else "#ef4444" if f['verdict'] == "weakness" else "#94a3b8"
            feature_bars_html += f"""
            <div class='feature-bar'>
                <div class='feature-labels'>
                    <span class='feature-name'>{f['label']}</span>
                    <span class='feature-val' style='color: {color}'>{f['value']:.2f}</span>
                </div>
                <div class='track'><div class='fill' style='width: {val_pct}%; background: {color};'></div></div>
            </div>
            """

        # Upskilling Recommendations
        upskill_html = ""
        if r['upskilling_recs']:
            lis = "".join([f"<li>{rec}</li>" for rec in r['upskilling_recs']])
            upskill_html = f"""
            <div class='section-title' style='margin-top:1.5rem;'>Actionable Upskilling</div>
            <div class='upskilling'><ul>{lis}</ul></div>
            """

        card = f"""
        <div class="card">
            <div class="card-header">
                <div>
                    <div class="match-title">{r['student_id']} → {r['job_id']}</div>
                    <div class="match-subtitle">Percentile: {r['percentile_in_cohort']:.2f} | Rank: {r['rank_in_cohort']}/{r['cohort_size']}</div>
                </div>
                {tier_html}
            </div>
            <div class="card-body">
                <div class="narrative">{r['detailed_narrative']}</div>
                
                <div class="section-title">Key Drivers</div>
                {feature_bars_html}

                <div class="section-title" style="margin-top:1.5rem;">Skill Coverage ({r['skill_coverage_pct']:.0%})</div>
                <div class="skills-grid">
                    <div>
                        <ul class="skill-list">{present_html}</ul>
                    </div>
                    <div>
                        <ul class="skill-list">{missing_html}</ul>
                    </div>
                </div>

                {upskill_html}
            </div>
        </div>
        """
        cards_html += card

    html = template.replace("{timestamp}", timestamp)
    html = html.replace("{total_matches}", str(len(results)))
    html = html.replace("{shortlisted_count}", str(shortlisted_count))
    html = html.replace("{cards_html}", cards_html)

    with open(HTML_OUT_PATH, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info(f"Admin Console saved -> {HTML_OUT_PATH}")

def run_pipeline():
    logger.info("=" * 60)
    logger.info("  PlaceMux Task 18 — Admin Console & Review Queue")
    logger.info("=" * 60)

    model = load_ranker_model()
    students, jobs = build_profiles()
    
    results = generate_enriched_explanations(model, students, jobs)

    if not results:
        raise RuntimeError("No explanations generated.")

    # Save raw JSON
    with open(REPORT_PATH, "w") as f:
        json.dump({"task": "Task 18", "matches": results}, f, indent=2)
    logger.info(f"Explainability Report saved -> {REPORT_PATH}")

    # Metrics
    metrics = {
        "task": "Task 18",
        "total_evaluations": len(results),
        "avg_match_score": sum(r["score"] for r in results) / len(results) if results else 0,
        "shortlisted": sum(1 for r in results if r["shortlist"])
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved -> {METRICS_PATH}")

    # Render HTML
    render_html_dashboard(results)

    logger.info("Task 18 pipeline complete. [OK]")

def main():
    try:
        run_pipeline()
    except Exception as e:
        logger.critical(f"Unhandled error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
