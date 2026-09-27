"""
Member 2 Matching & ML Execution Script for Business Entity Resolution Challenge.
Supports both real challenge dataset (6ab10eb3b23ba_student_resource/student_resource/dataset)
and synthetic sample dataset with configurable sample limits.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np

from src.preprocessing.normalize import preprocess_dataframe
from src.blocking.candidate_generator import CandidateGenerator
from src.features.pair_features import build_candidate_pair_features
from src.model.matching_model import MatchingModel, evaluate_pair_classifier
from src.evaluation.eval_matching import evaluate_macro_f05, optimize_threshold
from src.inference.predict_matches import generate_matching_predictions, save_matching_results
from utils.validate_submission import validate_matching_file

def load_real_dataset_sample(train_dir: str, num_s1: int = 1000):
    """
    Load a representative sample of real challenge data for fast, accurate training & validation.
    Reads num_s1 S1 entities, their ground truth matches, and candidate pools.
    """
    df_s1 = pd.read_csv(os.path.join(train_dir, "train_source1.tsv"), sep="\t", nrows=num_s1)
    s1_ids = set(df_s1["entity_id"])
    
    df_gt = pd.read_csv(os.path.join(train_dir, "train_ground_truth.tsv"), sep="\t")
    df_gt = df_gt[df_gt["source1_entity_id"].isin(s1_ids)].copy().reset_index(drop=True)
    
    target_matched_ids = set()
    for m in df_gt["matched_entity_ids"].dropna():
        for cid in str(m).split(","):
            if cid.strip():
                target_matched_ids.add(cid.strip())
                
    s2_path = os.path.join(train_dir, "train_source2.tsv")
    s3_path = os.path.join(train_dir, "train_source3.tsv")
    
    s2_rows = []
    s3_rows = []
    
    for chunk in pd.read_csv(s2_path, sep="\t", chunksize=100000):
        matched = chunk[chunk["entity_id"].isin(target_matched_ids)]
        if not matched.empty:
            s2_rows.append(matched)
        if len(s2_rows) > 0 and sum(len(c) for c in s2_rows) >= len(target_matched_ids):
            break

    for chunk in pd.read_csv(s3_path, sep="\t", chunksize=100000):
        matched = chunk[chunk["entity_id"].isin(target_matched_ids)]
        if not matched.empty:
            s3_rows.append(matched)
        if len(s3_rows) > 0 and sum(len(c) for c in s3_rows) >= len(target_matched_ids):
            break

    bg_s2 = pd.read_csv(s2_path, sep="\t", nrows=num_s1 * 3)
    bg_s3 = pd.read_csv(s3_path, sep="\t", nrows=num_s1 * 3)
    
    df_s2 = pd.concat(s2_rows + [bg_s2], ignore_index=True).drop_duplicates("entity_id")
    df_s3 = pd.concat(s3_rows + [bg_s3], ignore_index=True).drop_duplicates("entity_id")
        
    return df_s1, df_s2, df_s3, df_gt

def main():
    parser = argparse.ArgumentParser(description="Run Member 2 Matching & ML Pipeline")
    parser.add_argument("--num-entities", type=int, default=5000, help="Number of S1 entities to evaluate for real dataset run")
    parser.add_argument("--use-sample", action="store_true", help="Use 10-row synthetic sample dataset instead of real challenge data")
    args = parser.parse_args()

    print("=" * 70)
    print("       BUSINESS ENTITY RESOLUTION — MEMBER 2 MATCHING & ML")
    print("=" * 70)
    
    real_train_dir = "6ab10eb3b23ba_student_resource/student_resource/dataset/train"
    real_test_dir = "6ab10eb3b23ba_student_resource/student_resource/dataset/test"
    sample_train_dir = "dataset/train"
    sample_test_dir = "dataset/test"
    
    if not args.use_sample and os.path.exists(real_train_dir):
        train_dir = real_train_dir
        test_dir = real_test_dir
        is_real = True
        print(f" -> Mode: REAL CHALLENGE DATASET ({train_dir})")
    else:
        train_dir = sample_train_dir
        test_dir = sample_test_dir
        is_real = False
        print(f" -> Mode: SYNTHETIC SAMPLE DATASET ({train_dir})")

    output_dir = "output"
    models_dir = "models"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # ----------------------------------------------------
    # STEP 1: LOAD & PREPROCESS TRAINING DATA
    # ----------------------------------------------------
    print("\n[STEP 1] Loading and preprocessing training records...")
    if is_real:
        df_tr_s1, df_tr_s2, df_tr_s3, df_tr_gt = load_real_dataset_sample(train_dir, num_s1=args.num_entities)
    else:
        df_tr_s1 = pd.read_csv(os.path.join(train_dir, "train_source1.tsv"), sep="\t")
        df_tr_s2 = pd.read_csv(os.path.join(train_dir, "train_source2.tsv"), sep="\t")
        df_tr_s3 = pd.read_csv(os.path.join(train_dir, "train_source3.tsv"), sep="\t")
        df_tr_gt = pd.read_csv(os.path.join(train_dir, "train_ground_truth.tsv"), sep="\t")

    df_tr_s1 = preprocess_dataframe(df_tr_s1)
    df_tr_s2 = preprocess_dataframe(df_tr_s2)
    df_tr_s3 = preprocess_dataframe(df_tr_s3)
    
    df_tr_cands = pd.concat([df_tr_s2, df_tr_s3], ignore_index=True)
    print(f" -> Source 1: {len(df_tr_s1):,} | Candidate Pool (S2+S3): {len(df_tr_cands):,}")

    # ----------------------------------------------------
    # STEP 2: BLOCKING / CANDIDATE GENERATION (TRAIN)
    # ----------------------------------------------------
    print("\n[STEP 2] Generating candidate pairs via blocking...")
    generator = CandidateGenerator(min_token_len=3, max_candidates_per_s1=50, enable_fuzzy_fallback=True)
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
    
    print(f" -> Extracted {len(df_pairs):,} candidate pairs with {len(feature_names)} features each.")
    pos_count = int(df_pairs["label"].sum())
    neg_count = len(df_pairs) - pos_count
    print(f" -> Class balance: Positives = {pos_count:,}, Negatives = {neg_count:,} (Positive Ratio: {pos_count/len(df_pairs):.2%})")

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
    
    print(f" -> Train pairs: {len(X_train):,} (Pos: {int(y_train.sum()):,}) | Val pairs: {len(X_val):,} (Pos: {int(y_val.sum()):,})")

    # ----------------------------------------------------
    # STEP 5: MODEL TRAINING & BASELINE COMPARISON
    # ----------------------------------------------------
    print("\n[STEP 5] Training matching classifiers...")
    
    # 1. Baseline Logistic Regression
    lr_model = MatchingModel(model_type="logistic", random_state=42)
    lr_model.fit(X_train, y_train)
    lr_metrics = evaluate_pair_classifier(lr_model, X_val, y_val)
    auc_str_lr = f"{lr_metrics['pair_roc_auc']:.4f}" if not np.isnan(lr_metrics['pair_roc_auc']) else "N/A (Single class)"
    print(f" -> Logistic Regression (Pair ROC-AUC: {auc_str_lr}, Pair F0.5: {lr_metrics['pair_f05']:.4f})")
    
    # 2. HistGradientBoosting Tree Model
    gb_model = MatchingModel(model_type="hist_gb", random_state=42, max_iter=150)
    gb_model.fit(X_train, y_train)
    gb_metrics = evaluate_pair_classifier(gb_model, X_val, y_val)
    auc_str_gb = f"{gb_metrics['pair_roc_auc']:.4f}" if not np.isnan(gb_metrics['pair_roc_auc']) else "N/A (Single class)"
    print(f" -> HistGradientBoosting  (Pair ROC-AUC: {auc_str_gb}, Pair F0.5: {gb_metrics['pair_f05']:.4f})")

    # Select best model
    if not np.isnan(gb_metrics["pair_roc_auc"]) and not np.isnan(lr_metrics["pair_roc_auc"]):
        best_model = gb_model if gb_metrics["pair_roc_auc"] >= lr_metrics["pair_roc_auc"] else lr_model
    else:
        best_model = gb_model

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
    
    print("-" * 70)
    print(" THRESHOLD GRID SEARCH SUMMARY:")
    print(df_summary[["threshold", "macro_precision", "macro_recall", "macro_f05", "singleton_accuracy"]].to_string(index=False))
    print("-" * 70)
    print(f" OPTIMAL THRESHOLD        : {best_t:.2f}")
    print(f" MACRO F0.5 SCORE         : {best_metrics['macro_f05']:.4f}")
    print(f" MACRO PRECISION          : {best_metrics['macro_precision']:.4f}")
    print(f" MACRO RECALL             : {best_metrics['macro_recall']:.4f}")
    print(f" SINGLETON ACCURACY       : {best_metrics['singleton_accuracy']:.2%} ({best_metrics['singleton_correct']}/{best_metrics['singleton_total']})")
    print("-" * 70)

    # Re-train selected model on FULL training dataset sample
    print("\n[STEP 7] Training final model on 100% of training dataset & saving artifact...")
    final_model = MatchingModel(model_type=best_model.model_type, random_state=42)
    final_model.fit(df_features, df_pairs["label"])
    
    model_artifact_path = os.path.join(models_dir, "matching_model.joblib")
    final_model.save(model_artifact_path)

    # ----------------------------------------------------
    # STEP 8: TEST INFERENCE & MATCHING RESULTS
    # ----------------------------------------------------
    print("\n[STEP 8] Running test set inference and generating output/matching_results.tsv...")
    df_te_s1 = pd.read_csv(os.path.join(test_dir, "test_source1.tsv"), sep="\t")
    df_te_s2 = pd.read_csv(os.path.join(test_dir, "test_source2.tsv"), sep="\t")
    df_te_s3 = pd.read_csv(os.path.join(test_dir, "test_source3.tsv"), sep="\t")

    df_te_s1 = preprocess_dataframe(df_te_s1)
    df_te_s2 = preprocess_dataframe(df_te_s2)
    df_te_s3 = preprocess_dataframe(df_te_s3)
    
    df_te_cands = pd.concat([df_te_s2, df_te_s3], ignore_index=True)
    test_cand_dict = generator.generate_candidates(df_te_s1, df_te_cands)
    
    # Save candidate_pairs.tsv
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
