"""
Unit tests for Matching ML Model & Evaluation modules.
Tests model training, saving/loading, probability predictions, thresholding, and macro F0.5.
"""

import os
import tempfile
import numpy as np
import pandas as pd
from src.model.matching_model import MatchingModel, evaluate_pair_classifier
from src.evaluation.eval_matching import evaluate_macro_f05, optimize_threshold

def test_model_train_predict_save_load():
    X = pd.DataFrame({
        "name_lev_sim": [0.95, 0.90, 0.10, 0.05, 0.88, 0.20],
        "address_token_jaccard": [0.80, 0.85, 0.00, 0.10, 0.70, 0.05],
        "country_equal": [1.0, 1.0, 1.0, 0.0, 1.0, 1.0]
    })
    y = pd.Series([1, 1, 0, 0, 1, 0])
    
    # Test HistGradientBoosting
    model = MatchingModel(model_type="hist_gb", random_state=42)
    model.fit(X, y)
    
    probs = model.predict_proba(X)
    assert len(probs) == len(X)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    
    # Save & load artifact check
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.joblib")
        model.save(model_path)
        assert os.path.exists(model_path)
        
        loaded_model = MatchingModel.load(model_path)
        loaded_probs = loaded_model.predict_proba(X)
        np.testing.assert_allclose(probs, loaded_probs, rtol=1e-5)

def test_logistic_regression_model():
    X = pd.DataFrame({
        "feat1": [1.0, 0.8, 0.1, 0.0],
        "feat2": [0.9, 0.7, 0.0, 0.2]
    })
    y = pd.Series([1, 1, 0, 0])
    
    model = MatchingModel(model_type="logistic", random_state=42)
    model.fit(X, y)
    
    eval_metrics = evaluate_pair_classifier(model, X, y)
    assert eval_metrics["pair_precision"] > 0.5
    assert eval_metrics["pair_recall"] > 0.5

def test_macro_f05_and_threshold_optimizer():
    df_gt = pd.DataFrame([
        {"source1_entity_id": "S1-1", "matched_entity_ids": "S2-1"},
        {"source1_entity_id": "S1-2", "matched_entity_ids": ""},
        {"source1_entity_id": "S1-3", "matched_entity_ids": "S3-3"}
    ])
    
    df_pairs = pd.DataFrame([
        {"source1_entity_id": "S1-1", "candidate_entity_id": "S2-1"},
        {"source1_entity_id": "S1-1", "candidate_entity_id": "S2-99"},
        {"source1_entity_id": "S1-2", "candidate_entity_id": "S2-2"},
        {"source1_entity_id": "S1-3", "candidate_entity_id": "S3-3"}
    ])
    probs = np.array([0.90, 0.30, 0.20, 0.85])
    
    best_t, best_metrics, df_summary = optimize_threshold(
        df_pairs=df_pairs,
        probabilities=probs,
        df_ground_truth=df_gt,
        thresholds=[0.25, 0.50, 0.75]
    )
    
    assert 0.0 <= best_t <= 1.0
    assert "macro_f05" in best_metrics
    assert best_metrics["singleton_correct"] == 1
