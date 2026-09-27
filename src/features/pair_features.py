"""
Pairwise feature extraction module for Business Entity Resolution Challenge.
Computes complementary similarity metrics across business name, address, country,
missingness indicators, and interaction terms.
"""

import math
import re
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from rapidfuzz import fuzz, distance

def extract_tokens(text: str) -> List[str]:
    """Extract list of words from lowercased/normalized text."""
    if not isinstance(text, str) or not text.strip():
        return []
    return text.lower().split()

def extract_numeric_tokens(text: str) -> List[str]:
    """Extract numeric components (house numbers, suite numbers, zip codes)."""
    if not isinstance(text, str) or not text.strip():
        return []
    return re.findall(r"\d+", text)

def char_ngrams(text: str, n: int = 3) -> set:
    """Extract character n-grams from string."""
    if not text:
        return set()
    cleaned = f"^{text.strip()}$"
    if len(cleaned) < n:
        return {cleaned}
    return {cleaned[i:i+n] for i in range(len(cleaned) - n + 1)}

def jaccard_similarity(set1: set, set2: set) -> float:
    """Compute Jaccard index between two sets."""
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return float(intersection / union) if union > 0 else 0.0

def compute_pair_features(
    row_s1: Dict[str, Any],
    row_cand: Dict[str, Any]
) -> Dict[str, float]:
    """
    Compute dense numerical feature dictionary for a single candidate pair.
    
    :param row_s1: Dictionary/Series representing Source 1 record.
    :param row_cand: Dictionary/Series representing Candidate record (S2/S3).
    :return: Dictionary mapping feature_name -> float value.
    """
    raw_s1_name = str(row_s1.get("business_name", "") or "")
    raw_cand_name = str(row_cand.get("business_name", "") or "")
    
    norm_s1_name = str(row_s1.get("normalized_business_name", "") or raw_s1_name.lower())
    norm_cand_name = str(row_cand.get("normalized_business_name", "") or raw_cand_name.lower())
    
    can_s1_name = str(row_s1.get("canonical_business_name", "") or norm_s1_name)
    can_cand_name = str(row_cand.get("canonical_business_name", "") or norm_cand_name)
    
    norm_s1_addr = str(row_s1.get("normalized_business_address", "") or "")
    norm_cand_addr = str(row_cand.get("normalized_business_address", "") or "")
    
    norm_s1_country = str(row_s1.get("normalized_country", "") or "")
    norm_cand_country = str(row_cand.get("normalized_country", "") or "")
    
    cand_id = str(row_cand.get("entity_id", ""))
    
    # ------------------------------------------------------------------
    # 1. NAME SIMILARITY FEATURES
    # ------------------------------------------------------------------
    name_exact_raw = 1.0 if (raw_s1_name and raw_s1_name == raw_cand_name) else 0.0
    name_exact_norm = 1.0 if (norm_s1_name and norm_s1_name == norm_cand_name) else 0.0
    name_exact_canon = 1.0 if (can_s1_name and can_s1_name == can_cand_name) else 0.0
    
    # Normalized Levenshtein similarity (1.0 = identical, 0.0 = completely different)
    if norm_s1_name and norm_cand_name:
        lev_dist = distance.Levenshtein.distance(norm_s1_name, norm_cand_name)
        max_len = max(len(norm_s1_name), len(norm_cand_name))
        name_lev_sim = 1.0 - (lev_dist / max_len) if max_len > 0 else 1.0
        name_jw_sim = float(distance.JaroWinkler.similarity(norm_s1_name, norm_cand_name))
    else:
        name_lev_sim = 0.0
        name_jw_sim = 0.0
        
    tokens_s1_name = set(extract_tokens(can_s1_name or norm_s1_name))
    tokens_cand_name = set(extract_tokens(can_cand_name or norm_cand_name))
    
    name_token_jaccard = jaccard_similarity(tokens_s1_name, tokens_cand_name)
    name_token_overlap_count = float(len(tokens_s1_name.intersection(tokens_cand_name)))
    min_tokens_name = min(len(tokens_s1_name), len(tokens_cand_name))
    name_token_overlap_ratio = (name_token_overlap_count / min_tokens_name) if min_tokens_name > 0 else 0.0
    
    if norm_s1_name and norm_cand_name:
        name_token_set_ratio = float(fuzz.token_set_ratio(norm_s1_name, norm_cand_name) / 100.0)
        name_token_sort_ratio = float(fuzz.token_sort_ratio(norm_s1_name, norm_cand_name) / 100.0)
        name_partial_ratio = float(fuzz.partial_ratio(norm_s1_name, norm_cand_name) / 100.0)
    else:
        name_token_set_ratio = 0.0
        name_token_sort_ratio = 0.0
        name_partial_ratio = 0.0
        
    ngrams_s1_name = char_ngrams(norm_s1_name, n=3)
    ngrams_cand_name = char_ngrams(norm_cand_name, n=3)
    name_char_ngram_jaccard = jaccard_similarity(ngrams_s1_name, ngrams_cand_name)
    
    len_s1_n = len(norm_s1_name)
    len_cand_n = len(norm_cand_name)
    name_len_diff = float(abs(len_s1_n - len_cand_n))
    max_name_len = max(len_s1_n, len_cand_n)
    name_len_ratio = (min(len_s1_n, len_cand_n) / max_name_len) if max_name_len > 0 else 1.0
    name_token_count_diff = float(abs(len(tokens_s1_name) - len(tokens_cand_name)))

    # ------------------------------------------------------------------
    # 2. ADDRESS SIMILARITY FEATURES
    # ------------------------------------------------------------------
    address_exact_norm = 1.0 if (norm_s1_addr and norm_s1_addr == norm_cand_addr) else 0.0
    
    if norm_s1_addr and norm_cand_addr:
        addr_lev_dist = distance.Levenshtein.distance(norm_s1_addr, norm_cand_addr)
        max_addr_len = max(len(norm_s1_addr), len(norm_cand_addr))
        address_lev_sim = 1.0 - (addr_lev_dist / max_addr_len) if max_addr_len > 0 else 1.0
        address_token_set_ratio = float(fuzz.token_set_ratio(norm_s1_addr, norm_cand_addr) / 100.0)
    else:
        address_lev_sim = 0.0
        address_token_set_ratio = 0.0
        
    tokens_s1_addr = set(extract_tokens(norm_s1_addr))
    tokens_cand_addr = set(extract_tokens(norm_cand_addr))
    
    address_token_jaccard = jaccard_similarity(tokens_s1_addr, tokens_cand_addr)
    address_token_overlap_count = float(len(tokens_s1_addr.intersection(tokens_cand_addr)))
    min_tokens_addr = min(len(tokens_s1_addr), len(tokens_cand_addr))
    address_token_overlap_ratio = (address_token_overlap_count / min_tokens_addr) if min_tokens_addr > 0 else 0.0
    
    # Numeric tokens matching (house numbers, suite, zip codes)
    nums_s1 = set(extract_numeric_tokens(norm_s1_addr))
    nums_cand = set(extract_numeric_tokens(norm_cand_addr))
    if nums_s1 and nums_cand:
        num_overlap_count = float(len(nums_s1.intersection(nums_cand)))
        num_match_all = 1.0 if (nums_s1 == nums_cand) else 0.0
    elif not nums_s1 and not nums_cand:
        num_overlap_count = 0.0
        num_match_all = 1.0
    else:
        num_overlap_count = 0.0
        num_match_all = 0.0
        
    len_s1_a = len(norm_s1_addr)
    len_cand_a = len(norm_cand_addr)
    address_len_diff = float(abs(len_s1_a - len_cand_a))
    max_addr_l = max(len_s1_a, len_cand_a)
    address_len_ratio = (min(len_s1_a, len_cand_a) / max_addr_l) if max_addr_l > 0 else 1.0

    # ------------------------------------------------------------------
    # 3. COUNTRY FEATURES
    # ------------------------------------------------------------------
    if norm_s1_country and norm_cand_country:
        country_equal = 1.0 if norm_s1_country == norm_cand_country else 0.0
        country_missing = 0.0
    else:
        country_equal = 0.5  # Neutral indicator when country is missing
        country_missing = 1.0

    # ------------------------------------------------------------------
    # 4. MISSINGNESS FEATURES
    # ------------------------------------------------------------------
    name_missing_left = 1.0 if not norm_s1_name else 0.0
    name_missing_right = 1.0 if not norm_cand_name else 0.0
    address_missing_left = 1.0 if not norm_s1_addr else 0.0
    address_missing_right = 1.0 if not norm_cand_addr else 0.0
    address_missing_either = 1.0 if (not norm_s1_addr or not norm_cand_addr) else 0.0
    address_missing_both = 1.0 if (not norm_s1_addr and not norm_cand_addr) else 0.0

    # ------------------------------------------------------------------
    # 5. CROSS-FIELD INTERACTION FEATURES
    # ------------------------------------------------------------------
    name_high_addr_high = 1.0 if (name_token_set_ratio >= 0.8 and address_token_jaccard >= 0.5) else 0.0
    name_high_addr_low = 1.0 if (name_token_set_ratio >= 0.8 and address_token_jaccard < 0.3) else 0.0
    name_low_addr_high = 1.0 if (name_token_set_ratio < 0.4 and address_token_jaccard >= 0.6) else 0.0
    name_addr_sim_prod = float(name_token_set_ratio * address_token_jaccard)
    
    source_is_s2 = 1.0 if cand_id.startswith("S2-") else 0.0
    source_is_s3 = 1.0 if cand_id.startswith("S3-") else 0.0

    return {
        "name_exact_raw": name_exact_raw,
        "name_exact_norm": name_exact_norm,
        "name_exact_canon": name_exact_canon,
        "name_lev_sim": name_lev_sim,
        "name_jw_sim": name_jw_sim,
        "name_token_jaccard": name_token_jaccard,
        "name_token_overlap_count": name_token_overlap_count,
        "name_token_overlap_ratio": name_token_overlap_ratio,
        "name_token_set_ratio": name_token_set_ratio,
        "name_token_sort_ratio": name_token_sort_ratio,
        "name_partial_ratio": name_partial_ratio,
        "name_char_ngram_jaccard": name_char_ngram_jaccard,
        "name_len_diff": name_len_diff,
        "name_len_ratio": name_len_ratio,
        "name_token_count_diff": name_token_count_diff,
        "address_exact_norm": address_exact_norm,
        "address_lev_sim": address_lev_sim,
        "address_token_set_ratio": address_token_set_ratio,
        "address_token_jaccard": address_token_jaccard,
        "address_token_overlap_count": address_token_overlap_count,
        "address_token_overlap_ratio": address_token_overlap_ratio,
        "address_numeric_overlap": num_overlap_count,
        "address_numeric_match": num_match_all,
        "address_len_diff": address_len_diff,
        "address_len_ratio": address_len_ratio,
        "country_equal": country_equal,
        "country_missing": country_missing,
        "name_missing_left": name_missing_left,
        "name_missing_right": name_missing_right,
        "address_missing_left": address_missing_left,
        "address_missing_right": address_missing_right,
        "address_missing_either": address_missing_either,
        "address_missing_both": address_missing_both,
        "name_high_addr_high": name_high_addr_high,
        "name_high_addr_low": name_high_addr_low,
        "name_low_addr_high": name_low_addr_high,
        "name_addr_sim_prod": name_addr_sim_prod,
        "source_is_s2": source_is_s2,
        "source_is_s3": source_is_s3,
    }

def build_candidate_pair_features(
    df_s1: pd.DataFrame,
    df_candidates: pd.DataFrame,
    candidate_dict: Dict[str, List[str]],
    df_ground_truth: Optional[pd.DataFrame] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Build pair-level feature DataFrame for candidate generation output.
    
    :param df_s1: Source 1 preprocessed DataFrame indexed by entity_id.
    :param df_candidates: Candidate (S2+S3) preprocessed DataFrame indexed by entity_id.
    :param candidate_dict: Mapping s1_entity_id -> list of candidate entity_ids.
    :param df_ground_truth: Optional DataFrame with ground truth labels.
    :return: Tuple of (df_pairs, df_features, feature_names)
             df_pairs contains 'source1_entity_id', 'candidate_entity_id', and 'label' (if ground_truth given).
    """
    # Create fast dict lookups for records
    s1_dict = df_s1.set_index("entity_id").to_dict(orient="index")
    cand_dict = df_candidates.set_index("entity_id").to_dict(orient="index")
    
    # Ground truth set for label creation
    gt_map: Dict[str, set] = {}
    if df_ground_truth is not None:
        for _, row in df_ground_truth.iterrows():
            s1_id = row["source1_entity_id"]
            m_str = str(row["matched_entity_ids"]) if pd.notna(row["matched_entity_ids"]) else ""
            matches = set(m.strip() for m in m_str.split(",") if m.strip()) if m_str.strip() else set()
            gt_map[s1_id] = matches

    pairs = []
    feature_rows = []
    
    for s1_id, cand_list in candidate_dict.items():
        if s1_id not in s1_dict:
            continue
        s1_rec = s1_dict[s1_id]
        s1_rec["entity_id"] = s1_id
        
        true_matches_s1 = gt_map.get(s1_id, set())
        
        for cand_id in cand_list:
            if cand_id not in cand_dict:
                continue
            cand_rec = cand_dict[cand_id]
            cand_rec["entity_id"] = cand_id
            
            pair_entry = {
                "source1_entity_id": s1_id,
                "candidate_entity_id": cand_id
            }
            if df_ground_truth is not None:
                pair_entry["label"] = 1 if cand_id in true_matches_s1 else 0
                
            feats = compute_pair_features(s1_rec, cand_rec)
            
            pairs.append(pair_entry)
            feature_rows.append(feats)
            
    df_pairs = pd.DataFrame(pairs)
    df_features = pd.DataFrame(feature_rows)
    
    # Fill any potential NaNs or infs cleanly with 0.0
    df_features = df_features.fillna(0.0).replace([np.inf, -np.inf], 0.0)
    feature_names = list(df_features.columns) if not df_features.empty else []
    
    return df_pairs, df_features, feature_names
