"""
Submission Validation Tool for Business Entity Resolution Challenge.
Validates output/candidate_pairs.tsv and output/matching_results.tsv against test datasets.
"""

import argparse
import os
import sys
import pandas as pd
from typing import Set

def validate_candidate_file(candidate_path: str, test_dir: str) -> bool:
    """Validate candidate_pairs.tsv format and rules."""
    print(f"[*] Validating candidate file: {candidate_path}")
    if not os.path.exists(candidate_path):
        print(f"[ERROR] Candidate file does not exist: {candidate_path}")
        return False
        
    s1_path = os.path.join(test_dir, "test_source1.tsv")
    s2_path = os.path.join(test_dir, "test_source2.tsv")
    s3_path = os.path.join(test_dir, "test_source3.tsv")
    
    if not (os.path.exists(s1_path) and os.path.exists(s2_path) and os.path.exists(s3_path)):
        print(f"[ERROR] Missing test files in test directory: {test_dir}")
        return False
        
    df_s1 = pd.read_csv(s1_path, sep="\t")
    df_s2 = pd.read_csv(s2_path, sep="\t")
    df_s3 = pd.read_csv(s3_path, sep="\t")
    
    test_s1_ids: Set[str] = set(df_s1["entity_id"].astype(str))
    valid_cand_ids: Set[str] = set(df_s2["entity_id"].astype(str)).union(set(df_s3["entity_id"].astype(str)))
    
    try:
        df_cand = pd.read_csv(candidate_path, sep="\t", keep_default_na=False)
    except Exception as e:
        print(f"[ERROR] Failed to read candidate TSV file: {e}")
        return False
        
    required_cols = ["source1_entity_id", "candidate_entity_ids"]
    if list(df_cand.columns) != required_cols:
        print(f"[ERROR] Columns must be exactly {required_cols}, got {list(df_cand.columns)}")
        return False
        
    cand_s1_ids = df_cand["source1_entity_id"].astype(str).tolist()
    
    # Check 1: Row count & duplicate S1 IDs
    if len(cand_s1_ids) != len(set(cand_s1_ids)):
        print("[ERROR] Candidate file contains duplicate source1_entity_id entries!")
        return False
        
    cand_s1_set = set(cand_s1_ids)
    if cand_s1_set != test_s1_ids:
        missing = test_s1_ids - cand_s1_set
        extra = cand_s1_set - test_s1_ids
        if missing:
            print(f"[ERROR] Missing {len(missing)} test Source 1 entity IDs in candidate file.")
        if extra:
            print(f"[ERROR] Found {len(extra)} unexpected entity IDs in candidate file.")
        return False
        
    # Check 2: Candidate IDs validity & no duplicates per row
    invalid_cand_count = 0
    duplicate_cand_count = 0
    self_match_count = 0
    
    for _, row in df_cand.iterrows():
        s1_id = str(row["source1_entity_id"])
        cand_str = str(row["candidate_entity_ids"]).strip()
        
        if not cand_str:
            continue
            
        cands = [c.strip() for c in cand_str.split(",") if c.strip()]
        
        # Check duplicate candidates per row
        if len(cands) != len(set(cands)):
            duplicate_cand_count += 1
            
        for cid in cands:
            if cid == s1_id:
                self_match_count += 1
            if cid not in valid_cand_ids:
                invalid_cand_count += 1
                
    if duplicate_cand_count > 0:
        print(f"[ERROR] Found {duplicate_cand_count} rows with duplicate candidate IDs.")
        return False
        
    if self_match_count > 0:
        print(f"[ERROR] Found {self_match_count} instances where S1 entity matched itself.")
        return False
        
    if invalid_cand_count > 0:
        print(f"[ERROR] Found {invalid_cand_count} candidate IDs not present in test Source 2 or Source 3.")
        return False
        
    print("[SUCCESS] Candidate pairs file validation passed cleanly!")
    return True

def validate_matching_file(matching_path: str, candidate_path: str, test_dir: str) -> bool:
    """Validate matching_results.tsv format and integration rules."""
    print(f"[*] Validating matching results file: {matching_path}")
    if not os.path.exists(matching_path):
        print(f"[ERROR] Matching results file does not exist: {matching_path}")
        return False
        
    if not validate_candidate_file(candidate_path, test_dir):
        print("[ERROR] Candidate file validation failed before checking matching file.")
        return False
        
    df_cand = pd.read_csv(candidate_path, sep="\t", keep_default_na=False)
    cand_map = {}
    for _, row in df_cand.iterrows():
        s1 = str(row["source1_entity_id"])
        c_str = str(row["candidate_entity_ids"]).strip()
        cands = set(c.strip() for c in c_str.split(",") if c.strip()) if c_str else set()
        cand_map[s1] = cands
        
    df_match = pd.read_csv(matching_path, sep="\t", keep_default_na=False)
    required_cols = ["source1_entity_id", "matched_entity_ids"]
    if list(df_match.columns) != required_cols:
        print(f"[ERROR] Matching columns must be {required_cols}, got {list(df_match.columns)}")
        return False
        
    invalid_matches = 0
    for _, row in df_match.iterrows():
        s1 = str(row["source1_entity_id"])
        m_str = str(row["matched_entity_ids"]).strip()
        if not m_str:
            continue
        matches = [m.strip() for m in m_str.split(",") if m.strip()]
        allowed_cands = cand_map.get(s1, set())
        for m in matches:
            if m not in allowed_cands:
                invalid_matches += 1
                
    if invalid_matches > 0:
        print(f"[ERROR] Found {invalid_matches} matches not present in candidate_pairs.tsv!")
        return False
        
    print("[SUCCESS] Matching results file validation passed cleanly!")
    return True

def main():
    parser = argparse.ArgumentParser(description="Validate submission files for Amazon ML Challenge 2026")
    parser.add_argument("--candidate", type=str, required=True, help="Path to candidate_pairs.tsv")
    parser.add_argument("--matching", type=str, required=False, help="Path to matching_results.tsv (optional)")
    parser.add_argument("--test-dir", type=str, required=True, help="Path to dataset/test directory")
    
    args = parser.parse_args()
    
    cand_ok = validate_candidate_file(args.candidate, args.test_dir)
    if not cand_ok:
        sys.exit(1)
        
    if args.matching:
        match_ok = validate_matching_file(args.matching, args.candidate, args.test_dir)
        if not match_ok:
            sys.exit(1)
            
    print("\n[ALL CHECKS PASSED] Submission candidate file is valid.")
    sys.exit(0)

if __name__ == "__main__":
    main()
