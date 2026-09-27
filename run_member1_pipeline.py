"""
Member 1 Pipeline Execution Script for Amazon ML Challenge 2026.
Executes data preprocessing, normalization, blocking / candidate generation, 
candidate recall evaluation on training data, and generates output/candidate_pairs.tsv.
"""

import os
import sys
import pandas as pd

from src.preprocessing.normalize import preprocess_dataframe
from src.blocking.candidate_generator import CandidateGenerator
from src.blocking.eval_recall import evaluate_candidate_recall, print_recall_report
from utils.validate_submission import validate_candidate_file

def main():
    print("=" * 60)
    print("       AMAZON ML CHALLENGE 2026 — MEMBER 1 PIPELINE")
    print("=" * 60)
    
    train_dir = "dataset/train"
    test_dir = "dataset/test"
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # ----------------------------------------------------
    # STEP 1: TRAINING DATA INSPECTION & PREPROCESSING
    # ----------------------------------------------------
    print("\n[STEP 1] Loading and preprocessing training datasets...")
    train_s1_path = os.path.join(train_dir, "train_source1.tsv")
    train_s2_path = os.path.join(train_dir, "train_source2.tsv")
    train_s3_path = os.path.join(train_dir, "train_source3.tsv")
    train_gt_path = os.path.join(train_dir, "train_ground_truth.tsv")
    
    df_tr_s1 = pd.read_csv(train_s1_path, sep="\t")
    df_tr_s2 = pd.read_csv(train_s2_path, sep="\t")
    df_tr_s3 = pd.read_csv(train_s3_path, sep="\t")
    df_tr_gt = pd.read_csv(train_gt_path, sep="\t")
    
    print(f" -> Train Source 1 records: {len(df_tr_s1)}")
    print(f" -> Train Source 2 records: {len(df_tr_s2)}")
    print(f" -> Train Source 3 records: {len(df_tr_s3)}")
    print(f" -> Ground Truth records  : {len(df_tr_gt)}")
    
    df_tr_s1 = preprocess_dataframe(df_tr_s1)
    df_tr_s2 = preprocess_dataframe(df_tr_s2)
    df_tr_s3 = preprocess_dataframe(df_tr_s3)
    
    df_tr_cands = pd.concat([df_tr_s2, df_tr_s3], ignore_index=True)
    
    # ----------------------------------------------------
    # STEP 2 & 3: BLOCKING / CANDIDATE GENERATION ON TRAIN
    # ----------------------------------------------------
    print("\n[STEP 2 & 3] Running candidate generation on training dataset...")
    generator = CandidateGenerator(min_token_len=3, max_candidates_per_s1=100, enable_fuzzy_fallback=True)
    train_cand_dict = generator.generate_candidates(df_tr_s1, df_tr_cands)
    
    # ----------------------------------------------------
    # STEP 4: CANDIDATE RECALL EVALUATION
    # ----------------------------------------------------
    print("\n[STEP 4] Evaluating Candidate Recall against train_ground_truth.tsv...")
    recall_metrics = evaluate_candidate_recall(train_cand_dict, df_tr_gt)
    print_recall_report(recall_metrics)
    
    if recall_metrics["candidate_recall"] < 0.90:
        print("[WARNING] Candidate recall is below 90%. Consider loosening blocking constraints!")
        
    # ----------------------------------------------------
    # STEP 5: TEST CANDIDATE GENERATION & OUTPUT
    # ----------------------------------------------------
    print("\n[STEP 5] Generating candidate pairs for test dataset...")
    test_s1_path = os.path.join(test_dir, "test_source1.tsv")
    test_s2_path = os.path.join(test_dir, "test_source2.tsv")
    test_s3_path = os.path.join(test_dir, "test_source3.tsv")
    
    df_te_s1 = pd.read_csv(test_s1_path, sep="\t")
    df_te_s2 = pd.read_csv(test_s2_path, sep="\t")
    df_te_s3 = pd.read_csv(test_s3_path, sep="\t")
    
    print(f" -> Test Source 1 records: {len(df_te_s1)}")
    print(f" -> Test Source 2 records: {len(df_te_s2)}")
    print(f" -> Test Source 3 records: {len(df_te_s3)}")
    
    df_te_s1 = preprocess_dataframe(df_te_s1)
    df_te_s2 = preprocess_dataframe(df_te_s2)
    df_te_s3 = preprocess_dataframe(df_te_s3)
    
    df_te_cands = pd.concat([df_te_s2, df_te_s3], ignore_index=True)
    test_cand_dict = generator.generate_candidates(df_te_s1, df_te_cands)
    
    candidate_pairs_rows = []
    for _, row in df_te_s1.iterrows():
        s1_id = row["entity_id"]
        cands = test_cand_dict.get(s1_id, [])
        cand_str = ",".join(cands)
        candidate_pairs_rows.append({
            "source1_entity_id": s1_id,
            "candidate_entity_ids": cand_str
        })
        
    df_out = pd.DataFrame(candidate_pairs_rows)
    cand_pairs_path = os.path.join(output_dir, "candidate_pairs.tsv")
    df_out.to_csv(cand_pairs_path, sep="\t", index=False)
    print(f" -> Written candidate pairs to: {cand_pairs_path}")
    
    # ----------------------------------------------------
    # STEP 6: VALIDATE OUTPUT
    # ----------------------------------------------------
    print("\n[STEP 6] Validating candidate_pairs.tsv with submission validator...")
    is_valid = validate_candidate_file(cand_pairs_path, test_dir)
    
    if is_valid:
        print("\n[SUCCESS] Member 1 pipeline executed and validated successfully!")
    else:
        print("\n[ERROR] Output validation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
