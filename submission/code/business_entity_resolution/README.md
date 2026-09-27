# Business Entity Resolution

## 1. Project Overview
This project provides an end-to-end entity resolution pipeline for the Amazon ML Challenge — Business Entity Resolution task. Given reference entities from Source 1 (S1), the system identifies matching records from Source 2 (S2) and Source 3 (S3) corresponding to the same real-world business entity.

## 2. Directory Structure
```
code/business_entity_resolution/
├── src/
│   ├── preprocessing/      # Text & address normalization module
│   ├── blocking/           # Multi-key token index candidate generation module
│   ├── features/           # Pairwise string, token, and n-gram similarity extraction (39 features)
│   ├── model/              # Classifier models (HistGradientBoosting / LogisticRegression)
│   ├── evaluation/         # Entity-level Macro F0.5 metrics & threshold optimizer
│   └── inference/          # Test set prediction & submission formatter
├── README.md
└── requirements.txt
```

## 3. Environment
- **Python**: Version 3.10+ (tested on Python 3.11.9)
- **OS**: Windows / Linux / macOS compatible
- **Dependencies**: Listed in `requirements.txt`

## 4. Installation
Create and activate a virtual environment, then install required packages:

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
# source .venv/bin/activate

pip install -r requirements.txt
```

## 5. Input Data
Place the dataset TSV files in a dataset folder (e.g. `dataset/test` or `6ab10eb3b23ba_student_resource/student_resource/dataset/test`):
- `test_source1.tsv`
- `test_source2.tsv`
- `test_source3.tsv`

Optionally for training/validation:
- `train_source1.tsv`
- `train_source2.tsv`
- `train_source3.tsv`
- `train_ground_truth.tsv`

Do NOT modify column header formats.

## 6. Pipeline Architecture
```
Raw Business Data (S1, S2, S3)
  ↓
Preprocessing & Normalization (Accents, Legal Suffixes, Address Abbrev, Open-set Country)
  ↓
Multi-Index Blocking & Candidate Generation (Token & N-gram Hash Index, Fuzzy Fallback)
  ↓
Pairwise Feature Extraction (39 Name, Address, Token, N-gram, Country Match Features)
  ↓
Classifier Model (HistGradientBoosting / LogisticRegression with Probability Calibration)
  ↓
Probability Threshold Optimization (Entity-Level Macro F0.5 Search)
  ↓
Test Set Inference & Formatted Output (matching_results.tsv, candidate_pairs.tsv)
```

## 7. Running the Pipeline
To run the full end-to-end pipeline from the repository root:

```bash
python run_member2_pipeline.py
```

To run on sample test data for fast verification:
```bash
python run_member2_pipeline.py --use-sample
```

## 8. Output
Execution generates two tab-separated files in `output/`:
- `output/candidate_pairs.tsv`: Candidate match IDs per Source 1 entity
- `output/matching_results.tsv`: Final predicted matching entity IDs per Source 1 entity

## 9. Validation
Validate generated outputs against official formatting and constraint rules:

```bash
python utils/validate_submission.py --candidate output/candidate_pairs.tsv --matching output/matching_results.tsv --test-dir dataset/test
```

## 10. Reproducibility Notes
- **Random Seed**: Fixed at `42` across all splitting, training, and indexing operations.
- **Dependencies**: Pinned in `requirements.txt`.
- **Model Artifacts**: Trained model pipeline is serialized to `models/matching_model.joblib`.
- **No External Data/APIs**: Execution operates strictly on input dataset TSVs without external network calls.
