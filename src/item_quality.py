"""
item_quality.py — Psychometric evaluation engine for assessment items.

Provides the core logic for evaluating the quality of items in the test bank.
Calculates pass rates (difficulty), point-biserial correlation (discrimination),
and latency metrics. Flags items that fall outside acceptable thresholds.

Standing Instructions applied:
- NumPy style docstrings
- Defensive error handling
- Robust structured logging
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple

# ---------------------------------------------------------------------------
# Logging
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


def evaluate_item_bank(responses_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Evaluates item quality metrics across all items and flags weak ones.

    Groups the candidate response data by item to compute psychometric
    statistics. Correlates item correctness with the student's overall
    test score to assess item discrimination.

    Parameters
    ----------
    responses_df : pd.DataFrame
        Candidate responses containing columns:
        ['student_id', 'item_id', 'is_correct', 'latency_sec'].

    Returns
    -------
    Tuple[pd.DataFrame, List[Dict[str, Any]]]
        item_stats_df : DataFrame containing psychometric metrics per item.
        flagged_items : List of dictionaries for items failing quality thresholds.
    """
    try:
        # Validate input dataframe
        if responses_df is None or responses_df.empty:
            logger.warning("Empty response data provided. Returning empty evaluation.")
            return pd.DataFrame(), []
            
        required_cols = {'student_id', 'item_id', 'is_correct', 'latency_sec'}
        if not required_cols.issubset(responses_df.columns):
            raise ValueError(f"Input DataFrame must contain columns: {required_cols}")

        logger.info("Computing total scores per student for discrimination calculation...")
        # Compute overall score per student (sum of correct answers)
        student_scores = responses_df.groupby('student_id')['is_correct'].sum().reset_index()
        student_scores.rename(columns={'is_correct': 'total_score'}, inplace=True)
        
        # Merge back to get the total score on each response row
        merged_df = responses_df.merge(student_scores, on='student_id', how='left')
        
        logger.info("Calculating per-item psychometric metrics...")
        # We need to compute metrics per item
        item_stats = []
        for item_id, group in merged_df.groupby('item_id'):
            # 1. Difficulty (Pass Rate)
            pass_rate = float(group['is_correct'].mean())
            
            # 2. Discrimination (Point-Biserial Correlation)
            # Correlation between getting THIS item right (0/1) and TOTAL score
            # Guard against all 0s or all 1s which make variance 0 and correlation NaN
            if group['is_correct'].nunique() > 1 and group['total_score'].nunique() > 1:
                # Calculate Pearson correlation
                discrimination = float(np.corrcoef(group['is_correct'], group['total_score'])[0, 1])
                if np.isnan(discrimination) or np.isinf(discrimination):
                    discrimination = 0.0
            else:
                # If everyone passed or everyone failed, discrimination is undefined (0)
                discrimination = 0.0
                
            # 3. Latency
            median_latency = float(group['latency_sec'].median())
            
            # 4. Count
            response_count = int(len(group))
            
            item_stats.append({
                'item_id': item_id,
                'pass_rate': pass_rate,
                'discrimination': discrimination,
                'median_latency_sec': median_latency,
                'response_count': response_count
            })
            
        item_stats_df = pd.DataFrame(item_stats)
        
        logger.info("Flagging weak items based on quality thresholds...")
        # Evaluate thresholds
        flagged_items = flag_weak_items(item_stats_df)
        
        return item_stats_df, flagged_items

    except Exception as e:
        logger.error(f"Failed during item bank evaluation: {e}", exc_info=True)
        raise


def flag_weak_items(
    item_stats_df: pd.DataFrame,
    min_pass_rate: float = 0.20,
    max_pass_rate: float = 0.95,
    min_discrimination: float = 0.20,
    max_latency_sec: float = 120.0
) -> List[Dict[str, Any]]:
    """
    Identifies items that fail established psychometric quality thresholds.

    Parameters
    ----------
    item_stats_df : pd.DataFrame
        DataFrame of item metrics containing columns:
        ['item_id', 'pass_rate', 'discrimination', 'median_latency_sec']
    min_pass_rate : float
        Lower bound for pass rate. Items below are flagged TOO_HARD.
    max_pass_rate : float
        Upper bound for pass rate. Items above are flagged TOO_EASY.
    min_discrimination : float
        Lower bound for discrimination. Items below are LOW_DISCRIMINATION.
    max_latency_sec : float
        Upper bound for latency. Items above are HIGH_LATENCY.

    Returns
    -------
    List[Dict[str, Any]]
        List of dictionaries with keys: 'item_id', 'flag_reasons', 'metrics'.
    """
    if item_stats_df is None or item_stats_df.empty:
        return []
        
    flagged = []
    
    for _, row in item_stats_df.iterrows():
        reasons = []
        
        # Difficulty checks
        if row['pass_rate'] < min_pass_rate:
            reasons.append(f"TOO_HARD (Pass Rate: {row['pass_rate']:.1%})")
        elif row['pass_rate'] > max_pass_rate:
            reasons.append(f"TOO_EASY (Pass Rate: {row['pass_rate']:.1%})")
            
        # Discrimination check
        if row['discrimination'] < min_discrimination:
            reasons.append(f"LOW_DISCRIMINATION (r_pb: {row['discrimination']:.3f})")
            
        # Latency check
        if row['median_latency_sec'] > max_latency_sec:
            reasons.append(f"HIGH_LATENCY ({row['median_latency_sec']:.1f}s)")
            
        if reasons:
            flagged.append({
                'item_id': row['item_id'],
                'flag_reasons': reasons,
                'metrics': {
                    'pass_rate': row['pass_rate'],
                    'discrimination': row['discrimination'],
                    'median_latency_sec': row['median_latency_sec'],
                    'response_count': row['response_count']
                }
            })
            
    return flagged
