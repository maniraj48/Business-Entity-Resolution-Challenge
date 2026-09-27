"""
Member 2 Matching & ML Execution Script for Business Entity Resolution Challenge.
Executes pair feature extraction, entity-level validation split, model training (LR + HistGB),
threshold optimization for macro F0.5, model artifact saving, test inference, and output validation.
"""

import os
import sys
import pandas as pd
import numpy as np

from src.preprocessing.normalize import preprocess_dataframe
from src.blocking.candidate_generator import CandidateGenerator
from src.features.pair_features import build_candidate_pair_features
from src.model.matching_model import MatchingModel, evaluate_pair_classifier
from src.evaluation.eval_matching import evaluate_macro_f05, optimize_threshold
from src.inference.predict_matches import generate_matching_predictions, save_matching_results
from utils.validate_submission import validate_matching_file

def main():
    print("=" * 65)
    print("       BUSINESS ENTITY RESOLUTION — MEMBER 2 MATCHING & ML")
    print("=" * 65)
    
    # Check dataset location
    train_dir = "dataset/train"
    test_dir = "dataset/test"
    
    if not os.path.exists(train_dir):
        alt_train = "6ab10eb3b23ba_student_resource/student_resource/dataset/train"
        alt_test = "6ab10eb3b23ba_student_resource/student_resource/dataset/test"
        if os.path.exists(alt_train):
            train_dir = alt_train
            test_dir = alt_test
        else:
            print("[INFO] Generating sample dataset for pipeline execution...")
            from utils.generate_sample_datasets import generate_datasets
            generate_datasets()
            
    print(f" -> Using dataset paths:\n    Train: {train_dir}\n    Test:  {test_dir}")
    
    output_dir = "output"
    models_dir = "models"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # ----------------------------------------------------
    # STEP 1: LOAD & PREPROCESS TRAINING DATA
    # ----------------------------------------------------
    print("\n[STEP 1] Preprocessing training records...")
    df_tr_s1 = preprocess_dataframe(pd.read_csv(os.path.join(train_dir, "train_source1.tsv"), sep="\t"))
    df_tr_s2 = preprocess_dataframe(pd.read_csv(os.path.join(train_dir, "train_source2.tsv"), sep="\t"))
    df_tr_s3 = preprocess_dataframe(pd.read_csv(os.path.join(train_dir, "train_source3.tsv"), sep="\t"))
    df_tr_gt = pd.read_csv(os.path.join(train_dir, "train_ground_truth.tsv"), sep="\t")
    
    df_tr_cands = pd.concat([df_tr_s2, df_tr_s3], ignore_index=True)
    print(f" -> Source 1: {len(df_tr_s1)} | Candidate Pool (S2+S3): {len(df_tr_cands)}")

    # ----------------------------------------------------
    # STEP 2: BLOCKING / CANDIDATE GENERATION (TRAIN)
    # ----------------------------------------------------
    print("\n[STEP 2] Generating candidates for training dataset...")
    generator = CandidateGenerator(min_token_len=3, max_candidates_per_s1=100, enable_fuzzy_fallback=True)
    train_cand_dict = generator.generate_candidates(df_tr_s1, df_tr_cands)

    # ----------------------------------------------------
    # STEP 3: PAIR FEATURE EXTRACTION
    # ----------------------------------------------------
    print("\n[STEP 3] Extracting pair-level similarity features...")
    df_pairs, df_features, feature_names = build_candidate_pair_features(
        df_s1=df_tr_s1,
        df_candidates=df_tr_cands,
        candidate_dict=train_cand_dict,
        df_ground_truth=df_tr_gt
    )
    
    print(f" -> Extracted {len(df_pairs)} candidate pairs with {len(feature_names)} features each.")
    pos_count = int(df_pairs["label"].sum())
    neg_count = len(df_pairs) - pos_count
    print(f" -> Label distribution: Positive = {pos_count}, Negative = {neg_count} (Ratio: {pos_count/len(df_pairs):.2%})")

    # ----------------------------------------------------
    # STEP 4: ENTITY-LEVEL VALIDATION SPLIT
    # ----------------------------------------------------
    print("\n[STEP 4] Splitting train/validation sets by Source 1 entity (80/20)...")
    s1_ids = df_tr_s1["entity_id"].unique()
    np.random.seed(42)
    shuffled_s1 = np.random.permutation(s1_ids)
    split_idx = int(0.8 * len(shuffled_s1))
    
    train_s1_set = set(shuffled_s1[:split_idx])
    val_s1_set = set(shuffled_s1[split_idx:])
    
    train_mask = df_pairs["source1_entity_id"].isin(train_s1_set)
    val_mask = df_pairs["source1_entity_id"].isin(val_s1_set)
    
    X_train, y_train = df_features[train_mask], df_pairs[train_mask]["label"]
    X_val, y_val = df_features[val_mask], df_pairs[val_mask]["label"]
    
    df_val_pairs = df_pairs[val_mask].reset_index(drop=True)
    df_val_gt = df_tr_gt[df_tr_gt["source1_entity_id"].isin(val_s1_set)].reset_index(drop=True)
    
    print(f" -> Train pairs: {len(X_train)} | Val pairs: {len(X_val)}")

    # ----------------------------------------------------
    # STEP 5: MODEL TRAINING & BASELINE COMPARISON
    # ----------------------------------------------------
    print("\n[STEP 5] Training matching classifiers...")
    
    # 1. Baseline Logistic Regression
    lr_model = MatchingModel(model_type="logistic", random_state=42)
    lr_model.fit(X_train, y_train)
    lr_metrics = evaluate_pair_classifier(lr_model, X_val, y_val)
    print(f" -> Logistic Regression (Pair ROC-AUC: {lr_metrics['pair_roc_auc']:.4f}, Pair F0.5: {lr_metrics['pair_f05']:.4f})")
    
    # 2. HistGradientBoosting Tree Model
    gb_model = MatchingModel(model_type="hist_gb", random_state=42, max_iter=150)
    gb_model.fit(X_train, y_train)
    gb_metrics = evaluate_pair_classifier(gb_model, X_val, y_val)
    print(f" -> HistGradientBoosting  (Pair ROC-AUC: {gb_metrics['pair_roc_auc']:.4f}, Pair F0.5: {gb_metrics['pair_f05']:.4f})")

    # Select best model
    best_model = gb_model if gb_metrics["pair_roc_auc"] >= lr_metrics["pair_roc_auc"] else lr_model

    # ----------------------------------------------------
    # STEP 6: THRESHOLD OPTIMIZATION FOR MACRO F0.5
    # ----------------------------------------------------
    print("\n[STEP 6] Optimizing probability threshold for Entity-Level Macro F0.5...")
    val_probs = best_model.predict_proba(X_val)
    
    best_t, best_metrics, df_summary = optimize_threshold(
        df_pairs=df_val_pairs,
        probabilities=val_probs,
        df_ground_truth=df_val_gt
    )
    
    print("-" * 65)
    print(f" OPTIMAL THRESHOLD        : {best_t:.2f}")
    print(f" MACRO F0.5 SCORE         : {best_metrics['macro_f05']:.4f}")
    print(f" MACRO PRECISION          : {best_metrics['macro_precision']:.4f}")
    print(f" MACRO RECALL             : {best_metrics['macro_recall']:.4f}")
    print(f" SINGLETON ACCURACY       : {best_metrics['singleton_accuracy']:.2%} ({best_metrics['singleton_correct']}/{best_metrics['singleton_total']})")
    print("-" * 65)

    # Re-train selected model on FULL training dataset for maximum performance
    print("\n[STEP 7] Training final model on 100% of training data & saving artifact...")
    final_model = MatchingModel(model_type=best_model.model_type, random_state=42)
    final_model.fit(df_features, df_pairs["label"])
    
    model_artifact_path = os.path.join(models_dir, "matching_model.joblib")
    final_model.save(model_artifact_path)

    # ----------------------------------------------------
    # STEP 8: TEST INFERENCE & MATCHING RESULTS
    # ----------------------------------------------------
    print("\n[STEP 8] Running test set inference and generating output/matching_results.tsv...")
    df_te_s1 = preprocess_dataframe(pd.read_csv(os.path.join(test_dir, "test_source1.tsv"), sep="\t"))
    df_te_s2 = preprocess_dataframe(pd.read_csv(os.path.join(test_dir, "test_source2.tsv"), sep="\t"))
    df_te_s3 = preprocess_dataframe(pd.read_csv(os.path.join(test_dir, "test_source3.tsv"), sep="\t"))
    
    df_te_cands = pd.concat([df_te_s2, df_te_s3], ignore_index=True)
    test_cand_dict = generator.generate_candidates(df_te_s1, df_te_cands)
    
    # Save candidate_pairs.tsv if not present
    cand_pairs_path = os.path.join(output_dir, "candidate_pairs.tsv")
    cand_rows = []
    for _, r in df_te_s1.iterrows():
        s1 = r["entity_id"]
        cands = test_cand_dict.get(s1, [])
        cand_rows.append({"source1_entity_id": s1, "candidate_entity_ids": ",".join(cands)})
    pd.DataFrame(cand_rows).to_csv(cand_pairs_path, sep="\t", index=False)
    
    # Generate matching predictions with optimal threshold
    df_matching, df_scored_pairs = generate_matching_predictions(
        df_s1=df_te_s1,
        df_candidates=df_te_cands,
        candidate_dict=test_cand_dict,
        model=final_model,
        threshold=best_t
    )
    
    matching_path = os.path.join(output_dir, "matching_results.tsv")
    save_matching_results(df_matching, matching_path)

    # ----------------------------------------------------
    # STEP 9: SUBMISSION VALIDATION
    # ----------------------------------------------------
    print("\n[STEP 9] Validating matching_results.tsv against official rules...")
    is_valid = validate_matching_file(matching_path, cand_pairs_path, test_dir)
    
    if is_valid:
        print("\n[SUCCESS] Member 2 Matching & ML pipeline completed and validated cleanly!")
    else:
        print("\n[ERROR] Output validation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
