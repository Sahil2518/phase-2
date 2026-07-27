"""
train_task17.py - Task 17: Placement Dashboards & Recommendation v1 Live

Pipeline:
  1. Load ranker_v2 model
  2. Build 6-student x 5-job profiles
  3. Run Rec v1 (cohort recs + job shortlists)
  4. Save report/metrics JSON
  5. Plot heatmap PNG
  6. Generate placement dashboard HTML
  7. Run Python-level validation
  8. Save validation JSON

Standing instructions: robust error handling, structured logging,
NumPy-style docstrings, random_state=42.
"""

import os, sys, json, logging, datetime
import numpy as np

os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/task17.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

RANDOM_STATE    = 42
REPORT_PATH     = "logs/task17_rec_report.json"
METRICS_PATH    = "logs/task17_metrics.json"
HEATMAP_PATH    = "logs/task17_rec_heatmap.png"
DASHBOARD_PATH  = "logs/dashboard.html"
VALIDATION_PATH = "logs/task17_validation.json"
TOP_K_JOBS      = 3
TOP_K_CANDS     = 2

# ---------------------------------------------------------------------------
# Cohort Data — 6 students x 5 jobs
# ---------------------------------------------------------------------------
STUDENTS = [
    {"student_id":"STU-T17-001","skills_hard":["python","tensorflow","scikit-learn","sql","pandas","aws","docker"],"skills_soft":["analytical thinking","communication"],"years_experience":3.5,"education_level":3,"expected_salary":115000.0,"preferred_location":"Bangalore","remote_preference":"Hybrid","coding_score":0.88,"communication_score":0.78},
    {"student_id":"STU-T17-002","skills_hard":["javascript","react","vue.js","node.js","mongodb","rest api","git"],"skills_soft":["teamwork","creativity"],"years_experience":2.5,"education_level":2,"expected_salary":90000.0,"preferred_location":"Remote","remote_preference":"Remote","coding_score":0.75,"communication_score":0.80},
    {"student_id":"STU-T17-003","skills_hard":["aws","azure","docker","kubernetes","terraform","python","linux","bash"],"skills_soft":["leadership","documentation"],"years_experience":4.5,"education_level":2,"expected_salary":125000.0,"preferred_location":"Pune","remote_preference":"Hybrid","coding_score":0.83,"communication_score":0.85},
    {"student_id":"STU-T17-004","skills_hard":["kotlin","java","android","jetpack compose","room","retrofit","git"],"skills_soft":["attention to detail","adaptability"],"years_experience":2.0,"education_level":2,"expected_salary":80000.0,"preferred_location":"Mumbai","remote_preference":"On-site","coding_score":0.79,"communication_score":0.72},
    {"student_id":"STU-T17-005","skills_hard":["java","spring boot","postgresql","redis","rest api","docker","git"],"skills_soft":["collaboration","mentoring"],"years_experience":3.0,"education_level":2,"expected_salary":95000.0,"preferred_location":"Chennai","remote_preference":"Hybrid","coding_score":0.77,"communication_score":0.74},
    {"student_id":"STU-T17-006","skills_hard":["python","mlflow","docker","fastapi","git","sql","scikit-learn"],"skills_soft":["initiative","problem solving"],"years_experience":1.5,"education_level":3,"expected_salary":85000.0,"preferred_location":"Hyderabad","remote_preference":"Remote","coding_score":0.74,"communication_score":0.71},
]

JOBS = [
    {"job_id":"JOB-T17-001","required_skills":["python","tensorflow","scikit-learn","sql","docker"],"preferred_skills":["aws","mlflow"],"min_experience":3.0,"max_experience":None,"min_education":3,"salary_min":100000.0,"salary_max":140000.0,"job_location":"Bangalore","work_model":"Hybrid","min_coding_score":0.80,"min_communication_score":0.70},
    {"job_id":"JOB-T17-002","required_skills":["javascript","react","node.js","rest api","mongodb"],"preferred_skills":["typescript","vue.js"],"min_experience":2.0,"max_experience":None,"min_education":2,"salary_min":80000.0,"salary_max":110000.0,"job_location":"Remote","work_model":"Remote","min_coding_score":0.65,"min_communication_score":0.70},
    {"job_id":"JOB-T17-003","required_skills":["aws","docker","kubernetes","terraform","python"],"preferred_skills":["azure","jenkins","linux"],"min_experience":4.0,"max_experience":None,"min_education":2,"salary_min":110000.0,"salary_max":145000.0,"job_location":"Pune","work_model":"Hybrid","min_coding_score":0.78,"min_communication_score":0.78},
    {"job_id":"JOB-T17-004","required_skills":["kotlin","java","android","jetpack compose","rest api"],"preferred_skills":["room","retrofit","firebase"],"min_experience":1.5,"max_experience":None,"min_education":2,"salary_min":70000.0,"salary_max":95000.0,"job_location":"Mumbai","work_model":"On-site","min_coding_score":0.72,"min_communication_score":0.65},
    {"job_id":"JOB-T17-005","required_skills":["java","spring boot","postgresql","rest api","docker"],"preferred_skills":["redis","maven","kubernetes"],"min_experience":2.5,"max_experience":None,"min_education":2,"salary_min":85000.0,"salary_max":120000.0,"job_location":"Chennai","work_model":"Hybrid","min_coding_score":0.70,"min_communication_score":0.68},
]


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
    """Instantiate Pydantic StudentFeatures and JobFeatures from raw dicts."""
    from src.model_schemas import StudentFeatures, JobFeatures
    students = [StudentFeatures(**s) for s in STUDENTS]
    jobs     = [JobFeatures(**j)     for j in JOBS]
    logger.info(f"Built {len(students)} student profiles and {len(jobs)} job profiles.")
    return students, jobs


def run_recommendations(model, students, jobs):
    """Run cohort recommendations and job shortlists via Rec v1 engine."""
    from src.recommender import recommend_jobs_for_cohort, recommend_students_for_jobs
    logger.info("Running cohort recommendations...")
    cohort_rec = recommend_jobs_for_cohort(model=model, students=students, jobs=jobs, top_k=TOP_K_JOBS)
    logger.info("Running job shortlists...")
    shortlists = recommend_students_for_jobs(model=model, students=students, jobs=jobs, top_k=TOP_K_CANDS)
    return cohort_rec, shortlists


def save_report(students, jobs, cohort_rec, shortlists):
    """Generate Rec v1 report and save JSON + metrics."""
    from src.recommender import generate_rec_report
    report = generate_rec_report(students=students, jobs=jobs,
                                  cohort_recommendations=cohort_rec,
                                  job_shortlists=shortlists)
    report["task"] = "Task 17 - Rec v1 Live"
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Report saved -> {REPORT_PATH}")

    s = report["college_summary"]
    metrics = {
        "task": "Task 17 - Rec v1 Live",
        "timestamp": report["timestamp"],
        "college_cohort_size": report["college_cohort_size"],
        "jobs_evaluated": report["jobs_evaluated"],
        "avg_cohort_score": s["avg_cohort_score"],
        "placement_ready_count": s["placement_ready_count"],
        "placement_ready_pct": s["placement_ready_pct"],
        "top_skill_gap": s["top_skill_gap"],
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved -> {METRICS_PATH}")
    return report


def plot_heatmap(students, jobs, cohort_rec):
    """Save Student x Job match score heatmap PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        sids = [s.student_id for s in students]
        jids = [j.job_id     for j in jobs]
        matrix = np.zeros((len(sids), len(jids)))
        jidx = {jid: i for i, jid in enumerate(jids)}
        for rec in cohort_rec:
            row = sids.index(rec["student_id"])
            for tj in rec["top_jobs"]:
                col = jidx.get(tj["job_id"])
                if col is not None:
                    matrix[row][col] = tj["score"]

        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(matrix, cmap=plt.cm.RdYlGn, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(jids))); ax.set_xticklabels(jids, rotation=20, ha="right", fontsize=9)
        ax.set_yticks(range(len(sids))); ax.set_yticklabels(sids, fontsize=9)
        for i in range(len(sids)):
            for j in range(len(jids)):
                sc = matrix[i][j]
                c = "white" if (sc < 0.35 or sc > 0.65) else "black"
                ax.text(j, i, f"{sc:.2f}" if sc > 0 else "-", ha="center", va="center",
                        fontsize=9, fontweight="bold", color=c)
        plt.colorbar(im, ax=ax, label="Match Score")
        ax.set_title("Task 17 — Rec v1 Match Score Heatmap\nCollege Cohort × Job Pool",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(HEATMAP_PATH, dpi=150)
        plt.close()
        logger.info(f"Heatmap saved -> {HEATMAP_PATH}")
    except Exception as e:
        logger.warning(f"Heatmap generation failed (non-fatal): {e}")


def _score_color(score):
    """Return HSL background color for heatmap cell based on score."""
    hue = int(score * 120)
    return f"hsl({hue},70%,38%)"


def generate_dashboard(report, cohort_rec, shortlists, students, jobs):
    """
    Generate a self-contained HTML placement dashboard and save to logs/.

    Embeds all Rec v1 data inline — no external data file needed.

    Parameters
    ----------
    report : dict
    cohort_rec : list
    shortlists : list
    students : list[StudentFeatures]
    jobs : list[JobFeatures]
    """
    s = report["college_summary"]
    sids = [st.student_id for st in students]
    jids = [j.job_id      for j in jobs]

    # Build score matrix for heatmap table
    matrix = {sid: {jid: 0.0 for jid in jids} for sid in sids}
    for rec in cohort_rec:
        for tj in rec["top_jobs"]:
            matrix[rec["student_id"]][tj["job_id"]] = tj["score"]

    # Heatmap table rows
    heatmap_rows = ""
    for sid in sids:
        cells = ""
        for jid in jids:
            sc = matrix[sid][jid]
            bg = _score_color(sc) if sc > 0 else "#1e1e3a"
            tc = "white" if (sc < 0.35 or sc > 0.65) else "#111"
            disp = f"{sc:.2f}" if sc > 0 else "-"
            cells += f'<td style="background:{bg};color:{tc}">{disp}</td>'
        heatmap_rows += f"<tr><td class='row-label'>{sid}</td>{cells}</tr>"

    # Student recommendation cards
    rec_cards = ""
    for rec in cohort_rec:
        badge = "✅ Ready" if rec["placement_ready"] else "⚠️ Needs Work"
        badge_cls = "badge-ready" if rec["placement_ready"] else "badge-warn"
        jobs_html = ""
        for tj in rec["top_jobs"]:
            jobs_html += f"""<div class='rec-job'>
              <span class='rank'>#{tj['rank']}</span>
              <span class='jid'>{tj['job_id']}</span>
              <span class='score'>{tj['score']:.3f}</span>
            </div>"""
        rec_cards += f"""<div class='stu-card'>
          <div class='stu-header'>
            <span class='stu-id'>{rec['student_id']}</span>
            <span class='badge {badge_cls}'>{badge}</span>
          </div>
          <div class='stu-avg'>Avg Match: <b>{rec['avg_match_score']:.3f}</b></div>
          {jobs_html}
        </div>"""

    # Job shortlist table rows
    shortlist_rows = ""
    for jsl in shortlists:
        cands = ", ".join(
            f"{c['student_id']} ({c['score']:.3f})" for c in jsl["top_candidates"]
        )
        shortlist_rows += f"<tr><td>{jsl['job_id']}</td><td>{cands}</td><td>{jsl['avg_candidate_score']:.3f}</td></tr>"

    # Skill gap bars
    gap_detail = s.get("skill_gap_detail", {})
    max_gap = max(gap_detail.values(), default=1)
    gap_bars = ""
    for skill, count in sorted(gap_detail.items(), key=lambda x: -x[1])[:10]:
        pct = int(count / max_gap * 100)
        gap_bars += f"""<div class='gap-row'>
          <div class='gap-label'>{skill}</div>
          <div class='gap-track'><div class='gap-fill' style='width:{pct}%'>{count}</div></div>
        </div>"""

    # Column headers for heatmap
    col_headers = "".join(f"<th>{jid}</th>" for jid in jids)
    ts = report.get("timestamp", "")[:19].replace("T", " ")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PlaceMux Rec v1 — Placement Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Inter',sans-serif;background:#0d0d1a;background-image:radial-gradient(ellipse at top,#1a053380 0%,#0d0d1a 70%);color:#e2e8f0;min-height:100vh;padding:1.5rem}}
h2{{font-size:1.1rem;font-weight:600;color:#a78bfa;margin-bottom:1rem;letter-spacing:.04em;text-transform:uppercase}}
.header{{background:linear-gradient(135deg,#4f46e5,#7c3aed,#a855f7);border-radius:16px;padding:1.75rem 2rem;margin-bottom:1.5rem;position:relative;overflow:hidden}}
.header::after{{content:'';position:absolute;top:-60px;right:-60px;width:250px;height:250px;border-radius:50%;background:rgba(255,255,255,.07)}}
.header h1{{font-size:1.6rem;font-weight:700;color:#fff;position:relative}}
.header p{{color:rgba(255,255,255,.75);font-size:.85rem;margin-top:.3rem;position:relative}}
.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:1rem;margin-bottom:1.5rem}}
.stat{{background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.1);border-radius:12px;padding:1.25rem;text-align:center}}
.stat-val{{font-size:2rem;font-weight:700;color:#a78bfa}}
.stat-lbl{{font-size:.75rem;color:#94a3b8;margin-top:.3rem;text-transform:uppercase;letter-spacing:.05em}}
.card{{background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);border-radius:12px;padding:1.5rem;margin-bottom:1.5rem;backdrop-filter:blur(8px)}}
.heatmap-wrap{{overflow-x:auto}}
table.hm{{width:100%;border-collapse:collapse;font-size:.85rem}}
table.hm th{{background:rgba(124,58,237,.3);padding:.5rem .7rem;color:#a78bfa;font-size:.72rem;text-transform:uppercase;letter-spacing:.05em}}
table.hm td{{padding:.55rem .7rem;text-align:center;font-weight:600;border:1px solid rgba(255,255,255,.05);transition:transform .15s}}
table.hm td:hover{{transform:scale(1.1);z-index:1;position:relative;box-shadow:0 0 12px rgba(167,139,250,.4)}}
.row-label{{text-align:left!important;color:#94a3b8;font-size:.75rem;font-weight:500;background:#0d0d1a!important;white-space:nowrap}}
.recs-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:1rem}}
.stu-card{{background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);border-radius:10px;padding:1rem}}
.stu-header{{display:flex;justify-content:space-between;align-items:center;margin-bottom:.5rem}}
.stu-id{{font-weight:600;font-size:.85rem;color:#c4b5fd}}
.badge{{font-size:.7rem;padding:.2rem .6rem;border-radius:20px;font-weight:600}}
.badge-ready{{background:rgba(34,197,94,.2);color:#4ade80;border:1px solid rgba(34,197,94,.3)}}
.badge-warn{{background:rgba(245,158,11,.2);color:#fbbf24;border:1px solid rgba(245,158,11,.3)}}
.stu-avg{{font-size:.78rem;color:#94a3b8;margin-bottom:.6rem}}
.rec-job{{display:flex;gap:.5rem;align-items:center;padding:.3rem .5rem;border-radius:6px;background:rgba(255,255,255,.04);margin-bottom:.3rem;font-size:.8rem}}
.rank{{color:#a78bfa;font-weight:700;min-width:20px}}
.jid{{flex:1;color:#e2e8f0}}
.score{{color:#4ade80;font-weight:600}}
table.sl{{width:100%;border-collapse:collapse;font-size:.85rem}}
table.sl th{{background:rgba(124,58,237,.25);padding:.5rem .8rem;color:#a78bfa;text-align:left;font-size:.75rem;text-transform:uppercase}}
table.sl td{{padding:.5rem .8rem;border-bottom:1px solid rgba(255,255,255,.06);color:#cbd5e1}}
.gap-row{{display:flex;align-items:center;gap:.75rem;margin-bottom:.6rem}}
.gap-label{{width:120px;font-size:.8rem;color:#94a3b8;text-align:right;flex-shrink:0}}
.gap-track{{flex:1;background:rgba(255,255,255,.06);border-radius:6px;height:26px;overflow:hidden}}
.gap-fill{{height:100%;background:linear-gradient(90deg,#7c3aed,#4f46e5);display:flex;align-items:center;padding:0 .5rem;font-size:.75rem;color:#fff;font-weight:600;min-width:24px;border-radius:6px;transition:width .5s ease}}
.footer{{text-align:center;color:#475569;font-size:.75rem;margin-top:2rem}}
</style>
</head>
<body>
<div class="header">
  <h1>🎓 PlaceMux — Placement Dashboard</h1>
  <p>Recommendation v1 · Task 17 · Generated {ts}</p>
</div>

<div class="stats">
  <div class="stat"><div class="stat-val">{report['college_cohort_size']}</div><div class="stat-lbl">Students</div></div>
  <div class="stat"><div class="stat-val">{report['jobs_evaluated']}</div><div class="stat-lbl">Jobs Evaluated</div></div>
  <div class="stat"><div class="stat-val">{s['avg_cohort_score']:.3f}</div><div class="stat-lbl">Avg Match Score</div></div>
  <div class="stat"><div class="stat-val">{s['placement_ready_pct']:.0f}%</div><div class="stat-lbl">Placement Ready</div></div>
  <div class="stat"><div class="stat-val">{s['placement_ready_count']}/{report['college_cohort_size']}</div><div class="stat-lbl">Ready Count</div></div>
  <div class="stat"><div class="stat-val" style="font-size:1.1rem">{s['top_skill_gap']}</div><div class="stat-lbl">Top Skill Gap</div></div>
</div>

<div class="card">
  <h2>📊 Student × Job Match Heatmap</h2>
  <div class="heatmap-wrap">
  <table class="hm">
    <thead><tr><th>Student</th>{col_headers}</tr></thead>
    <tbody>{heatmap_rows}</tbody>
  </table>
  </div>
</div>

<div class="card">
  <h2>🏆 Top Recommendations per Student</h2>
  <div class="recs-grid">{rec_cards}</div>
</div>

<div class="card">
  <h2>📋 Job Shortlists</h2>
  <table class="sl">
    <thead><tr><th>Job</th><th>Top Candidates (score)</th><th>Avg Score</th></tr></thead>
    <tbody>{shortlist_rows}</tbody>
  </table>
</div>

<div class="card">
  <h2>🔍 Skill Gap Analysis</h2>
  <p style="font-size:.8rem;color:#64748b;margin-bottom:1rem">Required skills not covered by any student in the cohort</p>
  {gap_bars if gap_bars else '<p style="color:#4ade80">No skill gaps detected!</p>'}
</div>

<div class="footer">PlaceMux Phase 2 · Task 17 · Altrodav Technologies Pvt. Ltd.</div>
</body></html>"""

    with open(DASHBOARD_PATH, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info(f"Dashboard saved -> {DASHBOARD_PATH}")


def run_validation(report, cohort_rec, shortlists):
    """
    Validate pipeline outputs and API module import.

    Parameters
    ----------
    report : dict
    cohort_rec : list
    shortlists : list

    Returns
    -------
    dict  Validation results.
    """
    results = {}
    try:
        assert len(cohort_rec) == len(STUDENTS), "Not all students processed."
        results["cohort_rec_count"] = len(cohort_rec)
        results["cohort_rec"] = "PASS"
    except AssertionError as e:
        results["cohort_rec"] = f"FAIL: {e}"

    try:
        assert len(shortlists) == len(JOBS), "Not all jobs processed."
        results["shortlists_count"] = len(shortlists)
        results["shortlists"] = "PASS"
    except AssertionError as e:
        results["shortlists"] = f"FAIL: {e}"

    for path, key in [(REPORT_PATH,"report_file"),(METRICS_PATH,"metrics_file"),
                       (HEATMAP_PATH,"heatmap_file"),(DASHBOARD_PATH,"dashboard_file")]:
        results[key] = "PASS" if os.path.exists(path) else "FAIL: missing"

    try:
        import importlib, sys
        if "src.rec_api" in sys.modules:
            del sys.modules["src.rec_api"]
        importlib.import_module("src.rec_api")
        results["rec_api_import"] = "PASS"
    except Exception as e:
        results["rec_api_import"] = f"FAIL: {e}"

    avg = report.get("college_summary", {}).get("avg_cohort_score", 0)
    results["avg_cohort_score"] = avg
    results["score_sanity"] = "PASS" if 0 < avg < 1 else "WARN: unexpected score range"

    results["timestamp"] = datetime.datetime.now().isoformat()
    passed = sum(1 for v in results.values() if str(v).startswith("PASS"))
    total  = sum(1 for v in results.values() if str(v).startswith(("PASS","FAIL")))
    results["summary"] = f"{passed}/{total} checks passed"
    logger.info(f"Validation: {results['summary']}")

    with open(VALIDATION_PATH, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Validation saved -> {VALIDATION_PATH}")
    return results


def run_pipeline():
    """End-to-end Task 17 pipeline."""
    logger.info("=" * 60)
    logger.info("  PlaceMux Task 17 — Rec v1 Live Pipeline")
    logger.info(f"  Cohort: {len(STUDENTS)} students | Jobs: {len(JOBS)}")
    logger.info("=" * 60)

    model           = load_ranker_model()
    students, jobs  = build_profiles()
    cohort_rec, jsl = run_recommendations(model, students, jobs)

    if not cohort_rec:
        raise RuntimeError("No cohort recommendations produced — aborting.")

    report = save_report(students, jobs, cohort_rec, jsl)
    plot_heatmap(students, jobs, cohort_rec)
    generate_dashboard(report, cohort_rec, jsl, students, jobs)
    validation = run_validation(report, cohort_rec, jsl)

    logger.info("=" * 60)
    logger.info("  Task 17 pipeline complete. Rec v1 is LIVE. [OK]")
    logger.info(f"  Dashboard : {DASHBOARD_PATH}")
    logger.info(f"  API       : uvicorn src.rec_api:app --port 8001")
    logger.info(f"  Validation: {validation['summary']}")
    logger.info("=" * 60)


def main():
    try:
        run_pipeline()
    except FileNotFoundError as e:
        logger.critical(f"Missing required file: {e}")
        sys.exit(1)
    except RuntimeError as e:
        logger.critical(f"Runtime error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Unhandled fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
