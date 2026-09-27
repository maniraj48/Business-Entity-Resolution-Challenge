"""
Entity-level Macro F0.5 Evaluation Module for Business Entity Resolution Challenge.
Computes macro-averaged Precision, Recall, and F0.5 across all Source 1 entities,
including singletons, matching the exact competition evaluation formula.
"""

from typing import Dict, List, Set, Any, Tuple, Optional
import numpy as np
import pandas as pd

def compute_entity_f05(
    pred_matches: Set[str],
    true_matches: Set[str]
) -> Tuple[float, float, float]:
    """
    Compute Precision, Recall, and F0.5 score for a single Source 1 entity.
    
    :param pred_matches: Predicted set of candidate entity IDs.
    :param true_matches: Ground truth set of candidate entity IDs.
    :return: Tuple of (precision, recall, f05)
    """
    if not true_matches:
        # Singleton entity in ground truth
        if not pred_matches:
            return 1.0, 1.0, 1.0  # Correctly predicted singleton
        else:
            return 0.0, 0.0, 0.0  # False positive prediction on singleton

    if not pred_matches:
        # Ground truth has matches, but predicted empty
        return 0.0, 0.0, 0.0

    tp = len(pred_matches.intersection(true_matches))
    prec = tp / len(pred_matches)
    rec = tp / len(true_matches)
    
    denom = (0.25 * prec) + rec
    f05 = (1.25 * prec * rec) / denom if denom > 0 else 0.0
    return prec, rec, f05

def evaluate_macro_f05(
    predictions_map: Dict[str, List[str]],
    df_ground_truth: pd.DataFrame
) -> Dict[str, float]:
    """
    Evaluate macro-averaged entity-level precision, recall, and F0.5.
    
    :param predictions_map: Map s1_entity_id -> list of predicted candidate IDs.
    :param df_ground_truth: Ground truth DataFrame with columns 'source1_entity_id', 'matched_entity_ids'.
    :return: Dictionary containing macro_precision, macro_recall, macro_f05, singleton_accuracy, etc.
    """
    precisions = []
    recalls = []
    f05s = []
    
    singleton_total = 0
    singleton_correct = 0
    
    for _, row in df_ground_truth.iterrows():
        s1_id = row["source1_entity_id"]
        m_str = str(row["matched_entity_ids"]) if pd.notna(row["matched_entity_ids"]) else ""
        true_set = set(m.strip() for m in m_str.split(",") if m.strip()) if m_str.strip() else set()
        
        pred_set = set(predictions_map.get(s1_id, []))
        
        prec, rec, f05 = compute_entity_f05(pred_set, true_set)
        
        precisions.append(prec)
        recalls.append(rec)
        f05s.append(f05)
        
        if not true_set:
            singleton_total += 1
            if not pred_set:
                singleton_correct += 1
                
    macro_prec = float(np.mean(precisions)) if precisions else 0.0
    macro_rec = float(np.mean(recalls)) if recalls else 0.0
    macro_f05 = float(np.mean(f05s)) if f05s else 0.0
    singleton_acc = (singleton_correct / singleton_total) if singleton_total > 0 else 1.0
    
    return {
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f05": macro_f05,
        "singleton_total": singleton_total,
        "singleton_correct": singleton_correct,
        "singleton_accuracy": float(singleton_acc),
        "total_s1_entities": len(df_ground_truth)
    }

def optimize_threshold(
    df_pairs: pd.DataFrame,
    probabilities: np.ndarray,
    df_ground_truth: pd.DataFrame,
    thresholds: Optional[List[float]] = None
) -> Tuple[float, Dict[str, float], pd.DataFrame]:
    """
    Sweep across probability thresholds to find optimal threshold maximizing macro F0.5.
    
    :param df_pairs: DataFrame with columns 'source1_entity_id', 'candidate_entity_id'.
    :param probabilities: Array of predicted match probabilities for each row in df_pairs.
    :param df_ground_truth: DataFrame with ground truth labels.
    :param thresholds: Optional list of thresholds to evaluate.
    :return: Tuple of (best_threshold, best_metrics, df_results_summary)
    """
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.10, 0.95, 0.05)]
        
    df_eval_pairs = df_pairs[["source1_entity_id", "candidate_entity_id"]].copy()
    df_eval_pairs["prob"] = probabilities
    
    # Pre-group candidates by S1 entity
    s1_all_ids = list(df_ground_truth["source1_entity_id"].unique())
    grouped = df_eval_pairs.groupby("source1_entity_id")
    
    records = []
    best_threshold = 0.5
    best_f05 = -1.0
    best_metrics = {}
    
    for t in thresholds:
        # Build predictions map for threshold t
        pred_map: Dict[str, List[str]] = {s1_id: [] for s1_id in s1_all_ids}
        
        for s1_id, group in grouped:
            matches = group[group["prob"] >= t]["candidate_entity_id"].tolist()
            pred_map[s1_id] = matches
            
        metrics = evaluate_macro_f05(pred_map, df_ground_truth)
        metrics["threshold"] = t
        records.append(metrics)
        
        if metrics["macro_f05"] > best_f05:
            best_f05 = metrics["macro_f05"]
            best_threshold = t
            best_metrics = metrics
            
    df_summary = pd.DataFrame(records)
    return best_threshold, best_metrics, df_summary
