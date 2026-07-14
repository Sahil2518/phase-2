"""
train_task04.py — Task 04: Applications & Shortlisting — Explainability Pipeline

End-to-end pipeline:
  1. Load the trained ranker model from models/.
  2. Load sample students and jobs from data/.
  3. Rank jobs for each student (with explainability payloads).
  4. Rank candidates for each job (with explainability payloads).
  5. Save logs/task04_explainability.json — full results with explanation payloads.
  6. Save logs/task04_metrics.json — summary statistics.
  7. Generate logs/task04_contribution_chart.png — feature contribution bar chart.
  8. Print formatted explainability report to console.

Standing instructions: robust error handling, structured logging.
"""

import json
import os
import sys
import io
import logging
import datetime
import hashlib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Logging — must be configured before importing sibling modules
# ---------------------------------------------------------------------------
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

_stream_handler = logging.StreamHandler(
    stream=io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
)
_stream_handler.setFormatter(
    logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/task04.log", encoding="utf-8"),
        _stream_handler,
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from src.model_schemas import StudentFeatures, JobFeatures
from src.data_generator import generate_synthetic_data, FEATURE_COLS
from src.ranker import (
    train_ranker, save_ranker, load_ranker,
    rank_jobs_for_student, rank_candidates_for_job,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FEATURE_LABELS = {
    "skill_overlap_ratio":          "Skill Match",
    "norm_experience_gap":          "Experience Fit",
    "capped_salary_ratio":          "Salary Alignment",
    "education_met":                "Education Req.",
    "coding_threshold_met":         "Coding Prof.",
    "communication_threshold_met":  "Communication Prof.",
    "location_match":               "Location Compat.",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_json(filepath: str):
    """Load and return parsed JSON from a file."""
    if not os.path.exists(filepath):
        logger.critical(f"Required data file not found: {filepath}")
        sys.exit(1)
    with open(filepath, "r") as f:
        return json.load(f)


def print_explainability_report(title: str, ranked_items, entity_label: str = "ID"):
    """Pretty-print a ranked list with full explainability payloads."""
    print(f"\n{'=' * 78}")
    print(f"  {title}")
    print(f"{'=' * 78}")

    for rank, item in enumerate(ranked_items, start=1):
        payload = item.explanation_payload
        shortlist_icon = "[SHORTLISTED]" if payload and payload.shortlist else ""

        print(f"\n  #{rank}  {entity_label}: {item.id}  |  "
              f"Score: {item.score:.4f}  |  {shortlist_icon}")
        print(f"  {'-' * 72}")

        if payload:
            # Summary
            safe_summary = payload.summary.encode("ascii", errors="replace").decode("ascii")
            print(f"  Summary   : {safe_summary}")
            print(f"  Confidence: {payload.confidence.upper()}  "
                  f"({payload.factors_met}/{payload.factors_total} factors met)")
            print(f"  Shortlist : {'Yes' if payload.shortlist else 'No'} — {payload.shortlist_reason}")

            # Feature contributions table
            print(f"\n  {'Feature':<28} {'Value':>7} {'Contribution':>14} {'Verdict':>10}")
            print(f"  {'─' * 28} {'─' * 7} {'─' * 14} {'─' * 10}")
            for fc in payload.feature_contributions:
                verdict_icon = {"strength": "+", "weakness": "-", "neutral": "~"}.get(
                    fc.verdict, "?"
                )
                safe_label = fc.label.encode("ascii", errors="replace").decode("ascii")
                print(
                    f"  {safe_label:<28} {fc.value:>7.2f} {fc.contribution:>13.1%} "
                    f"  [{verdict_icon}]"
                )

            # Strengths / Weaknesses
            if payload.strengths:
                s_list = ", ".join(s.label for s in payload.strengths)
                safe_s = s_list.encode("ascii", errors="replace").decode("ascii")
                print(f"\n  Strengths  : {safe_s}")
            if payload.weaknesses:
                w_list = ", ".join(w.label for w in payload.weaknesses)
                safe_w = w_list.encode("ascii", errors="replace").decode("ascii")
                print(f"  Weaknesses : {safe_w}")
        else:
            # Fallback to legacy explanation
            for line in item.explanation:
                safe_line = line.encode("ascii", errors="replace").decode("ascii")
                print(f"    {safe_line}")

    print(f"\n{'=' * 78}\n")


# ---------------------------------------------------------------------------
# Visualisation — Feature Contribution Chart
# ---------------------------------------------------------------------------
def generate_contribution_chart(all_payloads, output_path: str):
    """
    Generate a grouped horizontal bar chart showing average feature
    contributions across all matches.
    """
    try:
        # Aggregate contributions
        contrib_sums = {feat: 0.0 for feat in FEATURE_COLS}
        contrib_counts = {feat: 0 for feat in FEATURE_COLS}

        for payload in all_payloads:
            for fc in payload.feature_contributions:
                if fc.feature in contrib_sums:
                    contrib_sums[fc.feature] += fc.contribution
                    contrib_counts[fc.feature] += 1

        # Compute averages
        features = []
        avg_contribs = []
        labels = []
        for feat in FEATURE_COLS:
            count = contrib_counts[feat]
            avg = contrib_sums[feat] / count if count > 0 else 0.0
            features.append(feat)
            avg_contribs.append(avg)
            labels.append(FEATURE_LABELS.get(feat, feat))

        # Sort descending by contribution
        sorted_indices = np.argsort(avg_contribs)[::-1]
        sorted_labels = [labels[i] for i in sorted_indices]
        sorted_contribs = [avg_contribs[i] for i in sorted_indices]

        # Colour by verdict threshold
        colors = []
        for i in sorted_indices:
            avg_val = contrib_sums[features[i]] / max(contrib_counts[features[i]], 1)
            if avg_val >= 0.15:
                colors.append("#2ecc71")  # green — high contribution
            elif avg_val >= 0.08:
                colors.append("#f39c12")  # amber — moderate
            else:
                colors.append("#e74c3c")  # red — low

        fig, ax = plt.subplots(figsize=(10, 5))
        y_pos = np.arange(len(sorted_labels))
        bars = ax.barh(y_pos, sorted_contribs, color=colors, edgecolor="#333", linewidth=0.5)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_labels, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel("Average Normalised Contribution", fontsize=11)
        ax.set_title(
            "Feature Contribution Analysis — All Matches\n"
            "(Task 04: Explainability)",
            fontsize=13, fontweight="bold", pad=14,
        )

        # Add value labels on bars
        for bar, val in zip(bars, sorted_contribs):
            ax.text(
                bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.1%}", va="center", fontsize=9, fontweight="bold",
            )

        # Legend
        legend_patches = [
            mpatches.Patch(color="#2ecc71", label="High impact"),
            mpatches.Patch(color="#f39c12", label="Moderate impact"),
            mpatches.Patch(color="#e74c3c", label="Low impact"),
        ]
        ax.legend(handles=legend_patches, loc="lower right", fontsize=9)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        logger.info(f"  -> Contribution chart saved to {output_path}")

    except Exception as e:
        logger.error(f"Contribution chart generation failed (non-fatal): {e}", exc_info=True)


# ---------------------------------------------------------------------------
# Shortlist summary visualisation
# ---------------------------------------------------------------------------
def generate_shortlist_heatmap(student_rankings, job_rankings, output_path: str):
    """
    Generate a shortlist heatmap: rows=students, cols=jobs.
    Cells are coloured by shortlist status.
    """
    try:
        # Build matrix from student rankings
        student_ids = []
        job_ids = []
        shortlist_data = {}

        for sr in student_rankings:
            sid = sr["student_id"]
            student_ids.append(sid)
            for rj in sr["ranked_jobs"]:
                jid = rj["id"]
                if jid not in job_ids:
                    job_ids.append(jid)
                payload = rj.get("explanation_payload", {})
                shortlist_data[(sid, jid)] = {
                    "score": rj["score"],
                    "shortlist": payload.get("shortlist", False) if payload else False,
                }

        # Build score matrix
        score_matrix = np.zeros((len(student_ids), len(job_ids)))
        annot_matrix = [[""]*len(job_ids) for _ in range(len(student_ids))]

        for i, sid in enumerate(student_ids):
            for j, jid in enumerate(job_ids):
                data = shortlist_data.get((sid, jid), {"score": 0, "shortlist": False})
                score_matrix[i][j] = data["score"]
                icon = "SL" if data["shortlist"] else ""
                annot_matrix[i][j] = f"{data['score']:.2f}\n{icon}"

        fig, ax = plt.subplots(figsize=(11, 6))

        # Custom colormap: red → yellow → green
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list(
            "shortlist", ["#e74c3c", "#f39c12", "#2ecc71"], N=256
        )

        import seaborn as sns
        sns.heatmap(
            score_matrix,
            annot=np.array(annot_matrix),
            fmt="",
            cmap=cmap,
            vmin=0.0, vmax=1.0,
            linewidths=1,
            linecolor="#ddd",
            xticklabels=job_ids,
            yticklabels=student_ids,
            ax=ax,
        )
        ax.set_title(
            "Shortlisting Matrix — Student x Job\n"
            "(SL = Shortlisted | Task 04: Explainability)",
            fontsize=13, fontweight="bold", pad=14,
        )
        ax.set_xlabel("Job ID", fontsize=11)
        ax.set_ylabel("Student ID", fontsize=11)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        logger.info(f"  -> Shortlist heatmap saved to {output_path}")

    except Exception as e:
        logger.error(f"Shortlist heatmap generation failed (non-fatal): {e}", exc_info=True)


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def run_pipeline():
    logger.info("=" * 70)
    logger.info("Starting Task 04 Pipeline: Applications & Shortlisting — Explainability")
    logger.info("=" * 70)

    # ── Step 1: Locate or Train the Ranker Model ─────────────────────────
    logger.info("[Step 1/8] Locating trained ranker model...")
    pkl_files = [
        f for f in os.listdir("models")
        if f.startswith("ranker_v1_") and f.endswith(".pkl")
    ]

    if pkl_files:
        pkl_files.sort(reverse=True)  # newest first
        pkl_path = os.path.join("models", pkl_files[0])
        logger.info(f"  -> Found existing model: {pkl_path}")
        model = load_ranker(pkl_path)
    else:
        logger.info("  -> No existing model found. Training a new ranker...")
        train_df = generate_synthetic_data(n_samples=600, random_state=42)
        model, metrics = train_ranker(train_df)
        pkl_path = save_ranker(model, metrics)
        model = load_ranker(pkl_path)

    # ── Step 2: Load Sample Data ─────────────────────────────────────────
    logger.info("[Step 2/8] Loading sample students and jobs...")
    try:
        students_data = load_json("data/sample_students.json")
        jobs_data     = load_json("data/sample_jobs.json")
        students = [StudentFeatures(**s) for s in students_data]
        jobs     = [JobFeatures(**j)     for j in jobs_data]
    except Exception as e:
        logger.critical(f"Data loading / validation failed: {e}", exc_info=True)
        sys.exit(1)

    logger.info(f"  -> {len(students)} students, {len(jobs)} jobs loaded.")

    # ── Step 3: Rank Jobs for Each Student (with Explainability) ─────────
    logger.info("[Step 3/8] Ranking jobs for each student with explainability...")
    student_rankings = []
    all_payloads = []

    for student in students:
        try:
            response = rank_jobs_for_student(model, student, jobs)
            student_rankings.append(response.model_dump())

            # Collect payloads for aggregation
            for rj in response.ranked_jobs:
                if rj.explanation_payload:
                    all_payloads.append(rj.explanation_payload)

            print_explainability_report(
                f"Jobs Ranked for Student: {student.student_id}",
                response.ranked_jobs,
                entity_label="Job",
            )
        except Exception as e:
            logger.error(f"  Failed for student {student.student_id}: {e}", exc_info=True)

    # ── Step 4: Rank Candidates for Each Job (with Explainability) ───────
    logger.info("[Step 4/8] Ranking candidates for each job with explainability...")
    job_rankings = []

    for job in jobs:
        try:
            response = rank_candidates_for_job(model, job, students)
            job_rankings.append(response.model_dump())

            for rc in response.ranked_candidates:
                if rc.explanation_payload:
                    all_payloads.append(rc.explanation_payload)

            print_explainability_report(
                f"Candidates Ranked for Job: {job.job_id}",
                response.ranked_candidates,
                entity_label="Student",
            )
        except Exception as e:
            logger.error(f"  Failed for job {job.job_id}: {e}", exc_info=True)

    # ── Step 5: Save Explainability JSON ─────────────────────────────────
    logger.info("[Step 5/8] Saving explainability results...")
    explainability_path = "logs/task04_explainability.json"
    output = {
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "task": "Task 04 — Applications & Shortlisting — Explainability",
        "model_artifact": pkl_path,
        "student_rankings": student_rankings,
        "job_rankings": job_rankings,
    }
    with open(explainability_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info(f"  -> Explainability results saved to {explainability_path}")

    # ── Step 6: Compute & Save Summary Metrics ───────────────────────────
    logger.info("[Step 6/8] Computing summary metrics...")
    total_matches = len(all_payloads)
    shortlisted = sum(1 for p in all_payloads if p.shortlist)
    confidence_dist = {"high": 0, "medium": 0, "low": 0}
    for p in all_payloads:
        confidence_dist[p.confidence] = confidence_dist.get(p.confidence, 0) + 1

    avg_score = np.mean([p.match_score for p in all_payloads]) if all_payloads else 0.0
    avg_factors_met = np.mean([p.factors_met for p in all_payloads]) if all_payloads else 0.0

    metrics = {
        "task": "Task 04",
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "total_matches_evaluated": total_matches,
        "shortlisted_count": shortlisted,
        "shortlist_rate": round(shortlisted / total_matches, 4) if total_matches > 0 else 0.0,
        "average_match_score": round(float(avg_score), 4),
        "average_factors_met": round(float(avg_factors_met), 2),
        "confidence_distribution": confidence_dist,
    }

    metrics_path = "logs/task04_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"  -> Metrics saved to {metrics_path}")

    # Print summary to console
    print(f"\n{'=' * 60}")
    print(f"  TASK 04 — EXPLAINABILITY SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Total matches evaluated : {total_matches}")
    print(f"  Shortlisted             : {shortlisted} ({metrics['shortlist_rate']:.0%})")
    print(f"  Average match score     : {avg_score:.4f}")
    print(f"  Average factors met     : {avg_factors_met:.1f} / 4")
    print(f"  Confidence distribution :")
    for band, count in confidence_dist.items():
        pct = count / total_matches * 100 if total_matches > 0 else 0
        print(f"    {band.upper():<8}: {count:>3} ({pct:.0f}%)")
    print(f"{'=' * 60}\n")

    # ── Step 7: Generate Visualisations ──────────────────────────────────
    logger.info("[Step 7/8] Generating visualisations...")

    chart_path = "logs/task04_contribution_chart.png"
    generate_contribution_chart(all_payloads, chart_path)

    heatmap_path = "logs/task04_shortlist_heatmap.png"
    generate_shortlist_heatmap(student_rankings, job_rankings, heatmap_path)

    # ── Step 8: Final Summary ────────────────────────────────────────────
    logger.info("[Step 8/8] Pipeline complete.")
    logger.info("=" * 70)
    logger.info("Task 04 Pipeline completed successfully.")
    logger.info(f"  Explainability JSON : {explainability_path}")
    logger.info(f"  Metrics JSON        : {metrics_path}")
    logger.info(f"  Contribution Chart  : {chart_path}")
    logger.info(f"  Shortlist Heatmap   : {heatmap_path}")
    logger.info("=" * 70)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        run_pipeline()
    except FileNotFoundError as e:
        logger.critical(f"Missing required file: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Unhandled fatal error: {e}", exc_info=True)
        sys.exit(1)
