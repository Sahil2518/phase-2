"""
train_task19.py — Bulk Onboarding & Recruiter Views (Item-bank quality)

Main execution script for Task 19. Synthesizes a dataset of candidate
responses across various assessment items, evaluates the quality of each
item in the bank using psychometric properties, and flags weak items for
administrator review.

Standing Instructions applied:
- NumPy style docstrings
- Defensive error handling
- Robust structured logging
- random_state=42 reproducibility
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd

from item_quality import evaluate_item_bank

# ---------------------------------------------------------------------------
# Logging & Setup
# ---------------------------------------------------------------------------
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/task19.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

np.random.seed(42)

def generate_synthetic_responses(num_students: int = 500, num_items: int = 50) -> pd.DataFrame:
    """
    Generates synthetic candidate response data to simulate an item bank.

    Creates intentional anomalies to test the item quality engine:
    - Item 10: Too hard (almost everyone fails)
    - Item 20: Too easy (almost everyone passes)
    - Item 30: High latency
    - Item 40: Low/negative discrimination (random responses)

    Parameters
    ----------
    num_students : int
        Number of synthetic students to generate.
    num_items : int
        Number of synthetic items in the bank.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['student_id', 'item_id', 'is_correct', 'latency_sec'].
    """
    logger.info(f"Generating synthetic responses for {num_students} students and {num_items} items.")
    
    # Assign true ability to each student (0 to 1)
    student_abilities = np.random.normal(loc=0.5, scale=0.15, size=num_students)
    student_abilities = np.clip(student_abilities, 0.05, 0.95)
    
    # Base difficulty per item (0 to 1, higher is easier)
    item_difficulties = np.random.uniform(0.3, 0.7, size=num_items)
    
    # Base latency per item (seconds)
    item_latencies = np.random.uniform(20, 90, size=num_items)
    
    data = []
    
    for s_idx in range(num_students):
        student_id = f"STU-{s_idx:04d}"
        ability = student_abilities[s_idx]
        
        for i_idx in range(num_items):
            item_id = f"ITEM-{i_idx:03d}"
            
            # Default behavior
            prob_correct = ability * item_difficulties[i_idx]
            latency_mu = item_latencies[i_idx]
            
            # Inject anomalies to test the flagging logic
            if i_idx == 10:  # Too hard
                prob_correct = 0.10
            elif i_idx == 20:  # Too easy
                prob_correct = 0.98
            elif i_idx == 30:  # High latency
                latency_mu = 150.0
            elif i_idx == 40:  # Low discrimination (pure noise, unaffected by ability)
                prob_correct = 0.50 
                
            is_correct = int(np.random.rand() < prob_correct)
            latency = max(1.0, np.random.normal(loc=latency_mu, scale=latency_mu * 0.2))
            
            data.append({
                'student_id': student_id,
                'item_id': item_id,
                'is_correct': is_correct,
                'latency_sec': latency
            })
            
    df = pd.DataFrame(data)
    logger.info(f"Generated {len(df)} total response records.")
    return df

def run_pipeline():
    """
    Executes the Task 19 end-to-end pipeline.
    
    1. Generates data.
    2. Runs item evaluation.
    3. Prints admin report.
    4. Saves metrics.
    """
    logger.info("--- Starting Task 19 Pipeline: Item-Bank Quality ---")
    
    # 1. Generate Data
    responses_df = generate_synthetic_responses(num_students=300, num_items=50)
    
    # Data Validation Guard
    assert responses_df.shape[0] > 0, "Response data is empty!"
    assert not responses_df.isnull().any().any(), "Response data contains NaNs."
    
    # 2. Evaluate Item Bank
    logger.info("Evaluating item bank for psychometric quality...")
    stats_df, flagged_items = evaluate_item_bank(responses_df)
    
    # 3. Print Admin Report
    print("\n" + "="*80)
    print(" 🚨 ADMIN REPORT: WEAK-ITEM FLAGS 🚨 ".center(80))
    print("="*80)
    
    if not flagged_items:
        print("✅ No weak items detected. Item bank is healthy.")
    else:
        print(f"Found {len(flagged_items)} items requiring review:\n")
        for item in flagged_items:
            reasons_str = " | ".join(item['flag_reasons'])
            print(f"► {item['item_id']} [{reasons_str}]")
            metrics = item['metrics']
            print(f"    Pass Rate: {metrics['pass_rate']:.1%} | "
                  f"Discrimination (r_pb): {metrics['discrimination']:.3f} | "
                  f"Median Latency: {metrics['median_latency_sec']:.1f}s")
            print("-" * 80)
            
    # 4. Save metrics
    metrics_path = "logs/task19_item_metrics.json"
    logger.info(f"Saving comprehensive item statistics to {metrics_path}")
    
    output_data = {
        "summary": {
            "total_items": len(stats_df),
            "flagged_items_count": len(flagged_items)
        },
        "flagged_items": flagged_items,
        "all_item_stats": stats_df.to_dict(orient='records')
    }
    
    with open(metrics_path, "w") as f:
        json.dump(output_data, f, indent=4)
        
    logger.info("--- Task 19 Pipeline Complete ---")

def main():
    """
    Main entry point with fatal error trap.
    """
    try:
        run_pipeline()
    except Exception as e:
        logger.critical(f"Unhandled fatal error in pipeline: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
