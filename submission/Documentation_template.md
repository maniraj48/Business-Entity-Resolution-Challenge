# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Entity Resolution Squad  
**Team Members:** Member 1 (Preprocessing & Blocking), Member 2 (ML & Matching Model)  
**Submission Date:** September 2026

---

## 1. Executive Summary
We developed an end-to-end entity resolution pipeline for identifying matching business entities across Source 1 (reference set), Source 2, and Source 3. Our system combines multi-index token and character n-gram candidate blocking with a 39-feature pairwise classification model (HistGradientBoosting) optimized directly for entity-level Macro F0.5.

---

## 2. Methodology

### 2.1 Problem Analysis
During exploratory data analysis, we identified key noise patterns and challenges:
- **Legal Suffix Variations**: High frequency of corporate abbreviations (Inc, LLC, Corp, GmbH, S.A., Ltd) causing false mismatches in standard string comparison.
- **Address Noise**: Variations in street types (Street vs St, Avenue vs Ave, Road vs Rd), suite numbers, and missing zip codes.
- **Open-Set Country Names**: Non-standard country representations (e.g., "United States", "USA", "US", "U.S.A.").
- **Class Imbalance & Singletons**: A significant portion of Source 1 entities have zero matching records in Source 2/3 (singletons), requiring strict probability calibration to prevent false positives.

### 2.2 Solution Strategy

**Approach Type:** Multi-Index Blocking + 39-Feature Pairwise HistGradientBoosting Classifier  
**Core Innovation:** Entity-level Macro F0.5 threshold optimization coupled with multi-key token/n-gram indexing and fuzzy fallback to balance high candidate recall with precision.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:** 
  1. Significant name tokens (length ≥ 3, excluding stop words and legal suffixes).
  2. Character 3-gram name prefixes.
  3. Country-partitioned fuzzy fallback matching when token indexing yields insufficient candidates.
- **Candidate pairs generated:** Maximum 50 candidates per Source 1 entity (configurable).
- **How true matches were not lost:** Multi-key indexing combines token overlap indices with fuzzy ratio fallback, ensuring high recall across variations in word order and typos while keeping candidate count manageable.

---

## 4. Matching Model

**Features used (39 pairwise similarity features):**
- **Name features:** Levenshtein distance, RapidFuzz ratio, partial_ratio, token_sort_ratio, token_set_ratio, WRatio, character n-gram Jaccard similarity (n=2,3,4), prefix/suffix match indicators.
- **Address features:** Address string Levenshtein distance, token overlap ratio, token Jaccard similarity, digit sequence match, street number match indicator.
- **Other:** Country exact match boolean indicator, length ratio of names/addresses.

**Model type:** HistGradientBoostingClassifier (with Logistic Regression baseline comparison) enclosed in a scikit-learn Pipeline with StandardScaler.  
**Threshold selection method:** Grid search over probability thresholds [0.10, 0.90] evaluating entity-level Macro F0.5 on an 80/20 entity-wise validation split.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** Evaluated on validation split during threshold grid search; optimal threshold selected based on highest Macro F0.5 score.
- **Common false positives (wrong merges):** Co-located businesses sharing identical street addresses or generic brand names with different franchise identifiers.
- **Common false negatives (missed matches):** Severely truncated business names combined with missing or misspelled street addresses.

---

## 6. Conclusion
Our entity resolution pipeline effectively bridges string-level noise and entity-level matching through structured text normalization, multi-key candidate blocking, and 39 pairwise similarity features fed into a calibrated gradient boosted classifier. The system is lightweight, self-contained, fully reproducible, and compliant with all challenge evaluation metrics.

---

## Appendix

### A. Code Artefacts
The complete runnable code is packaged in `code/business_entity_resolution/`:
- `src/preprocessing/normalize.py`: Text, legal suffix, address, and country normalization.
- `src/blocking/candidate_generator.py`: Multi-key candidate pair blocking.
- `src/features/pair_features.py`: 39-feature pair similarity generator.
- `src/model/matching_model.py`: Classifier model definition & evaluation.
- `src/evaluation/eval_matching.py`: Entity-level Macro F0.5 and threshold optimization.
- `src/inference/predict_matches.py`: Test prediction generator & output formatting.

**Pipeline Entrypoint:** `python run_member2_pipeline.py`  
**Outputs Generated:** `output/matching_results.tsv` and `output/candidate_pairs.tsv`

### B. Additional Results
Threshold search output validates singleton accuracy (entities with zero matches correctly predicted as empty match lists) while maximizing precision-heavy Macro F0.5.
