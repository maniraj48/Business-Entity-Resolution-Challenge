"""
Matching Inference & Submission Generator Module for Business Entity Resolution Challenge.
Generates candidate pair predictions and builds formatted matching_results.tsv.
"""

import os
import pandas as pd
from typing import Dict, List, Optional, Tuple
from src.features.pair_features import build_candidate_pair_features
from src.model.matching_model import MatchingModel

def generate_matching_predictions(
    df_s1: pd.DataFrame,
    df_candidates: pd.DataFrame,
    candidate_dict: Dict[str, List[str]],
    model: MatchingModel,
    threshold: float = 0.5
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate match predictions for a dataset given preprocessed records and candidate dict.
    
    :param df_s1: Preprocessed Source 1 DataFrame.
    :param df_candidates: Preprocessed Candidate DataFrame (S2+S3).
    :param candidate_dict: Map s1_entity_id -> list of candidate entity_ids.
    :param model: Trained MatchingModel instance.
    :param threshold: Classification probability threshold.
    :return: Tuple of (df_matching_results, df_scored_pairs)
    """
    # 1. Extract pair features
    df_pairs, df_features, feature_names = build_candidate_pair_features(
        df_s1=df_s1,
        df_candidates=df_candidates,
        candidate_dict=candidate_dict,
        df_ground_truth=None
    )
    
    match_map: Dict[str, List[str]] = {s1_id: [] for s1_id in df_s1["entity_id"].unique()}
    
    if not df_pairs.empty and not df_features.empty:
        # Align feature columns with model's trained features
        for f_col in model.feature_names:
            if f_col not in df_features.columns:
                df_features[f_col] = 0.0
        df_features_aligned = df_features[model.feature_names]
        
        # Predict probabilities
        probs = model.predict_proba(df_features_aligned)
        df_pairs["match_probability"] = probs
        
        # Filter predictions above threshold
        df_matched = df_pairs[df_pairs["match_probability"] >= threshold]
        
        for _, row in df_matched.iterrows():
            s1_id = str(row["source1_entity_id"])
            cand_id = str(row["candidate_entity_id"])
            if s1_id in match_map:
                match_map[s1_id].append(cand_id)
                
    # Build final formatted results DataFrame
    matching_rows = []
    for s1_id in df_s1["entity_id"].unique():
        matched_cands = match_map.get(s1_id, [])
        match_str = ",".join(matched_cands)
        matching_rows.append({
            "source1_entity_id": s1_id,
            "matched_entity_ids": match_str
        })
        
    df_matching_results = pd.DataFrame(matching_rows)
    return df_matching_results, df_pairs

def save_matching_results(df_matching: pd.DataFrame, output_path: str) -> None:
    """Save matching predictions to tab-separated output file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_matching.to_csv(output_path, sep="\t", index=False)
    print(f" -> Written matching results to: {output_path}")
