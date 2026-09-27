"""
Candidate Recall Evaluation module for Business Entity Resolution Challenge.
Evaluates recall of candidate generator on training ground truth labels.
"""

from typing import Dict, List, Set, Any
import numpy as np
import pandas as pd

def evaluate_candidate_recall(
    candidate_dict: Dict[str, List[str]],
    df_ground_truth: pd.DataFrame
) -> Dict[str, Any]:
    """
    Evaluate candidate recall against ground truth dataset.
    
    :param candidate_dict: Mapping s1_entity_id -> list of candidate entity_ids.
    :param df_ground_truth: Dataframe loaded from train_ground_truth.tsv
                            (columns: 'source1_entity_id', 'matched_entity_ids').
    :return: Dictionary containing all recall metrics and candidate stats.
    """
    total_s1 = len(df_ground_truth)
    total_true_matches = 0
    captured_true_matches = 0
    
    candidates_per_s1 = []
    zero_candidate_s1_count = 0
    
    for _, row in df_ground_truth.iterrows():
        s1_id = row["source1_entity_id"]
        matched_str = str(row["matched_entity_ids"]) if pd.notna(row["matched_entity_ids"]) else ""
        
        # Parse ground truth true match set
        true_matches: Set[str] = set()
        if matched_str.strip():
            true_matches = set(m.strip() for m in matched_str.split(",") if m.strip())
            
        total_true_matches += len(true_matches)
        
        cands = set(candidate_dict.get(s1_id, []))
        cand_count = len(cands)
        candidates_per_s1.append(cand_count)
        
        if cand_count == 0:
            zero_candidate_s1_count += 1
            
        if true_matches:
            # Count how many ground truth matches were included in generated candidate set
            captured = len(true_matches.intersection(cands))
            captured_true_matches += captured
            
    recall = (captured_true_matches / total_true_matches) if total_true_matches > 0 else 1.0
    avg_cands = float(np.mean(candidates_per_s1)) if candidates_per_s1 else 0.0
    median_cands = float(np.median(candidates_per_s1)) if candidates_per_s1 else 0.0
    max_cands = int(np.max(candidates_per_s1)) if candidates_per_s1 else 0
    
    results = {
        "total_s1_entities": total_s1,
        "total_true_matches": total_true_matches,
        "captured_true_matches": captured_true_matches,
        "candidate_recall": recall,
        "candidate_recall_pct": float(recall * 100.0),
        "avg_candidates_per_s1": avg_cands,
        "median_candidates_per_s1": median_cands,
        "max_candidates_per_s1": max_cands,
        "zero_candidate_s1_count": zero_candidate_s1_count
    }
    return results

def print_recall_report(metrics: Dict[str, Any]) -> None:
    """Print formatted candidate recall evaluation report."""
    print("=" * 60)
    print("         CANDIDATE RECALL EVALUATION REPORT")
    print("=" * 60)
    print(f" Total Source 1 Entities : {metrics['total_s1_entities']}")
    print(f" Total True Match Pairs  : {metrics['total_true_matches']}")
    print(f" Captured True Matches   : {metrics['captured_true_matches']}")
    print(f" Candidate Recall        : {metrics['candidate_recall_pct']:.2f}%")
    print("-" * 60)
    print(f" Avg Candidates per S1   : {metrics['avg_candidates_per_s1']:.2f}")
    print(f" Median Candidates per S1: {metrics['median_candidates_per_s1']:.1f}")
    print(f" Max Candidates per S1   : {metrics['max_candidates_per_s1']}")
    print(f" S1 with Zero Candidates : {metrics['zero_candidate_s1_count']}")
    print("=" * 60)
