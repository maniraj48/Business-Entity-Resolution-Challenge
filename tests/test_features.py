"""
Unit tests for Pairwise Feature Engineering module.
Tests exact matches, typos, missing values, country comparison, NaN checks, and feature shapes.
"""

import numpy as np
import pandas as pd
from src.features.pair_features import compute_pair_features, build_candidate_pair_features

def test_exact_matching_pair():
    rec1 = {
        "business_name": "Acme Corporation",
        "normalized_business_name": "acme corporation",
        "canonical_business_name": "acme",
        "business_address": "100 Main St, New York, NY",
        "normalized_business_address": "100 main street new york ny",
        "country": "US",
        "normalized_country": "united states"
    }
    rec2 = rec1.copy()
    rec2["entity_id"] = "S2-001"
    
    feats = compute_pair_features(rec1, rec2)
    assert feats["name_exact_norm"] == 1.0
    assert feats["name_exact_canon"] == 1.0
    assert feats["address_exact_norm"] == 1.0
    assert feats["country_equal"] == 1.0
    assert feats["name_lev_sim"] == 1.0
    assert feats["name_token_jaccard"] == 1.0

def test_similar_and_different_names():
    rec_base = {
        "business_name": "Apex Technologies Inc",
        "normalized_business_name": "apex technologies inc",
        "canonical_business_name": "apex",
        "business_address": "500 Innovation Way",
        "normalized_business_address": "500 innovation way",
        "country": "US",
        "normalized_country": "united states"
    }
    rec_similar = {
        "entity_id": "S2-002",
        "business_name": "Apex Tech Corp",
        "normalized_business_name": "apex tech corp",
        "canonical_business_name": "apex",
        "business_address": "500 Innovation Way Ste 10",
        "normalized_business_address": "500 innovation way suite 10",
        "country": "US",
        "normalized_country": "united states"
    }
    rec_diff = {
        "entity_id": "S2-003",
        "business_name": "Blue Sky Bakery",
        "normalized_business_name": "blue sky bakery",
        "canonical_business_name": "blue sky bakery",
        "business_address": "12 Baker St",
        "normalized_business_address": "12 baker street",
        "country": "US",
        "normalized_country": "united states"
    }
    
    feats_sim = compute_pair_features(rec_base, rec_similar)
    feats_diff = compute_pair_features(rec_base, rec_diff)
    
    assert feats_sim["name_token_set_ratio"] > feats_diff["name_token_set_ratio"]
    assert feats_sim["name_exact_canon"] == 1.0
    assert feats_diff["name_token_jaccard"] == 0.0

def test_missing_address_and_name():
    rec1 = {
        "business_name": "Test Entity",
        "normalized_business_name": "test entity",
        "canonical_business_name": "test entity",
        "business_address": "",
        "normalized_business_address": "",
        "country": "France",
        "normalized_country": "france"
    }
    rec2 = {
        "entity_id": "S3-100",
        "business_name": "",
        "normalized_business_name": "",
        "canonical_business_name": "",
        "business_address": "10 Rue de Rivoli",
        "normalized_business_address": "10 rue de rivoli",
        "country": "France",
        "normalized_country": "france"
    }
    
    feats = compute_pair_features(rec1, rec2)
    assert feats["address_missing_left"] == 1.0
    assert feats["name_missing_right"] == 1.0
    assert feats["address_missing_either"] == 1.0
    assert feats["country_equal"] == 1.0

def test_candidate_pair_features_build():
    df_s1 = pd.DataFrame([
        {"entity_id": "S1-1", "business_name": "Acme", "normalized_business_name": "acme", "business_address": "Main", "normalized_business_address": "main", "country": "US", "normalized_country": "united states"}
    ])
    df_cands = pd.DataFrame([
        {"entity_id": "S2-1", "business_name": "Acme Inc", "normalized_business_name": "acme inc", "business_address": "Main St", "normalized_business_address": "main street", "country": "US", "normalized_country": "united states"},
        {"entity_id": "S3-1", "business_name": "Random Co", "normalized_business_name": "random co", "business_address": "Pine St", "normalized_business_address": "pine street", "country": "US", "normalized_country": "united states"}
    ])
    cand_dict = {"S1-1": ["S2-1", "S3-1"]}
    df_gt = pd.DataFrame([{"source1_entity_id": "S1-1", "matched_entity_ids": "S2-1"}])
    
    df_pairs, df_features, feat_names = build_candidate_pair_features(
        df_s1, df_cands, cand_dict, df_gt
    )
    
    assert len(df_pairs) == 2
    assert len(df_features) == 2
    assert "label" in df_pairs.columns
    assert df_pairs.iloc[0]["label"] == 1
    assert df_pairs.iloc[1]["label"] == 0
    assert not df_features.isnull().values.any()
    assert not np.isinf(df_features.values).any()
