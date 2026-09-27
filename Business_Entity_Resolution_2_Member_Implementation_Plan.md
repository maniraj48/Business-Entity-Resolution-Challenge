# Amazon ML Challenge 2026 — 2-Member Implementation Plan

## 1. Team Setup

The team has exactly two members and uses **only the `main` Git branch**.

### Member 1 — Candidate Generation

Responsible for:

- Data inspection
- Name/address normalization
- Blocking
- Candidate generation
- Candidate recall evaluation
- `output/candidate_pairs.tsv`

### Member 2 — Matching Model

Responsible for:

- Pairwise feature engineering
- Matching model
- Validation
- Threshold optimization
- Singleton handling
- Final inference
- `output/matching_results.tsv`

### Fixed pipeline

```text
Source 1 + Source 2 + Source 3
              |
              v
       Normalization
              |
              v
          Blocking
              |
              v
     candidate_pairs.tsv
              |
              v
      Pairwise Features
              |
              v
       Matching Model
              |
              v
   Threshold + Singleton Logic
              |
              v
    matching_results.tsv
```

---

# 2. Shared Project Structure

```text
Business-Entity-Resolution-Challenge/
│
├── dataset/
│
├── src/
│   ├── preprocessing/      # Member 1
│   ├── blocking/           # Member 1
│   ├── features/           # Member 2
│   ├── model/              # Member 2
│   ├── evaluation/         # Member 2
│   └── inference/          # Member 2
│
├── output/
│   ├── candidate_pairs.tsv
│   └── matching_results.tsv
│
├── experiments/
│
├── README.md
├── requirements.txt
└── run_pipeline.py
```

Do not randomly change this architecture.

---

# 3. Implementation Order

```text
PHASE 0 — Common setup
        ↓
PHASE 1 — Data inspection
        ↓
PHASE 2 — Member 1 normalization
        ↓
PHASE 3 — Member 1 blocking
        ↓
PHASE 4 — Candidate recall checkpoint
        ↓
PHASE 5 — Member 2 pair features
        ↓
PHASE 6 — Member 2 matching model
        ↓
PHASE 7 — Validation + F0.5
        ↓
PHASE 8 — Test prediction
        ↓
PHASE 9 — Submission validation
        ↓
PHASE 10 — Final review
```

Do not skip the candidate-recall checkpoint.

---

# 4. PHASE 0 — Common Setup

Both members must first inspect:

```text
README.md
dataset/
existing src/
requirements.txt
problem statement
challenge guidelines
```

Both must agree on:

```text
S1 = reference source
S2 = candidate source
S3 = candidate source
```

All TSV files must be read with:

```python
pd.read_csv(path, sep="\t")
```

Do not assume comma-separated files.

---

# 5. PHASE 1 — Data Inspection

Inspect only the information needed for implementation:

- row counts
- columns
- missing values
- country distribution
- duplicate values
- business-name patterns
- address patterns

Avoid unnecessary charts and analysis.

Create a small summary if useful:

```text
data_summary.md
```

Do not invent statistics. Use only values actually measured from the dataset.

---

# 6. PHASE 2 — Member 1: Normalization

Create:

```text
src/preprocessing/normalize.py
```

Suggested functions:

```text
normalize_name()
normalize_address()
normalize_country()
```

Preserve original fields.

Use separate normalized fields such as:

```text
business_name
normalized_business_name

business_address
normalized_business_address
```

Reasonable normalization may include:

- lowercase
- whitespace normalization
- punctuation normalization
- `&` / `and`
- common legal suffix variations
- safe common abbreviations

Do not aggressively normalize values if that could merge different businesses.

### Country requirement

Treat country as an open-set string.

Do not hard-code the country list to US/India because France appears in the test set.

---

# 7. PHASE 3 — Member 1: Blocking

Create reusable code under:

```text
src/blocking/
```

Investigate multiple lightweight candidate-generation signals:

1. Exact normalized name
2. Name token overlap
3. Address token overlap
4. Character/n-gram similarity where computationally reasonable
5. Country as a supporting signal

Primary objective:

> High candidate recall with a reasonably small candidate set.

Do not optimize final F0.5 at this stage.

---

# 8. PHASE 4 — Candidate Recall Checkpoint

Before Member 2 builds the final model, Member 1 must evaluate the candidate set using:

```text
dataset/train/train_ground_truth.tsv
```

Measure:

- candidate recall
- average candidates per S1
- median candidates per S1
- maximum candidates per S1
- S1 entities with zero candidates

Conceptually:

```text
candidate recall =
true matches captured by candidates
------------------------------------
total true matches
```

If candidate recall is poor, improve blocking before proceeding.

---

# 9. Member 1 Output

Generate:

```text
output/candidate_pairs.tsv
```

Required columns:

```text
source1_entity_id
candidate_entity_ids
```

Example:

```text
source1_entity_id    candidate_entity_ids
S1-00001             S2-00047,S2-00193,S3-00812
S1-00002             S3-00004
S1-00003
```

Rules:

- exactly one row for every Source 1 entity
- candidates must only be S2/S3 IDs
- no duplicate candidate IDs
- empty candidate lists are allowed
- candidate IDs must exist in the relevant test source
- this is the final candidate set that Member 2 scores
- do not put final matching decisions here

---

# 10. PHASE 5 — Member 2: Pair Features

Only start after `candidate_pairs.tsv` is stable.

Create:

```text
src/features/pair_features.py
```

For every candidate pair, investigate compact features.

### Name

- exact normalized name
- token overlap
- Jaccard similarity
- edit/Levenshtein similarity
- character n-gram TF-IDF cosine
- length difference

### Address

- exact normalized address
- token overlap
- Jaccard similarity
- edit/Levenshtein similarity
- character n-gram TF-IDF cosine
- length difference

### Country

- exact country match

Do not create hundreds of redundant features.

Reuse Member 1's normalization logic where appropriate.

---

# 11. PHASE 6 — Member 2: Matching Model

Create training labels using:

```text
dataset/train/train_ground_truth.tsv
```

For each candidate pair:

```text
positive = true match
negative = not a true match
```

Prefer hard negatives from the candidate set.

Start with lightweight models:

1. Logistic Regression
2. Random Forest
3. Gradient Boosting / XGBoost if appropriate

Do not test many models without evidence of improvement.

---

# 12. PHASE 7 — Validation + F0.5

Create a validation split from the training data.

Measure:

- precision
- recall
- macro F0.5
- false merges
- missed matches
- singleton behavior

The official metric is macro-averaged F0.5.

Do not select the model using accuracy alone.

### Threshold testing

Test a reasonable range such as:

```text
0.50
0.55
0.60
0.65
0.70
0.75
0.80
0.85
0.90
```

Select the threshold using validation macro F0.5.

Do not assume `0.5` is the correct threshold.

---

# 13. Singleton Handling

Do not force every Source 1 entity to have a match.

If all candidate scores are weak, allow:

```text
source1_entity_id    matched_entity_ids
S1-xxxxx
```

with an empty match list.

Correct singleton detection matters.

---

# 14. PHASE 8 — Final Test Prediction

Use the selected model and threshold on:

```text
output/candidate_pairs.tsv
```

Generate:

```text
output/matching_results.tsv
```

Required columns:

```text
source1_entity_id
matched_entity_ids
```

Rules:

- exactly one row for every test Source 1 entity
- matched IDs only from S2/S3
- no duplicate IDs
- empty list allowed
- every final match must appear in `candidate_pairs.tsv`

---

# 15. PHASE 9 — Submission Validation

Run:

```bash
python utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

Do not submit until the validator passes.

---

# 16. PHASE 10 — Final Review

Both members verify:

```text
[ ] No external business data used
[ ] No external entity lookup used
[ ] No geocoding API used
[ ] No S1 → S1 matches
[ ] No duplicate IDs
[ ] Every test S1 exists
[ ] Every final match exists in candidate_pairs.tsv
[ ] Empty matches are allowed
[ ] candidate_pairs.tsv is the actual model input
[ ] matching_results.tsv has the exact required format
[ ] Validator passes
[ ] Code can reproduce the outputs
[ ] requirements.txt is sufficient
```

---

# 17. Git Workflow — Main Only

No branches.

Before working:

```bash
git checkout main
git pull origin main
```

After tested work:

```bash
git add .
git commit -m "Short meaningful message"
git push origin main
```

Recommended sequence:

```text
Member 1
  ↓
implement
  ↓
test
  ↓
push main
  ↓
Member 2 pulls
  ↓
implement
  ↓
test
  ↓
push main
```

Do not push broken code.

Do not use:

```text
git push --force
git reset --hard
git clean -fd
```

without explicit human approval.

---

# 18. AI Anti-Hallucination Protocol

Both AI coding agents MUST follow:

```text
INSPECT
   ↓
PLAN
   ↓
IMPLEMENT
   ↓
TEST
   ↓
REPORT
```

## Before modifying code

Inspect:

- existing files
- existing functions
- imports
- actual dataset structure
- README
- requirements

Do not invent:

- filenames
- columns
- functions
- metrics
- dataset properties
- statistics

## Before implementation

Briefly state:

```text
FILES TO MODIFY:
FILES TO CREATE:
IMPLEMENTATION PLAN:
FILES THAT MUST NOT BE MODIFIED:
```

Then implement only the assigned responsibility.

## After implementation

Run the relevant tests.

Never claim:

> "This works"

unless it was actually executed.

Never claim a model improved unless the improvement was actually measured.

Never fabricate precision, recall, F0.5, candidate recall, or dataset statistics.

If something is unknown:

> Inspect it or explicitly state that it is unknown.

---

# 19. Required AI Completion Report

Every AI agent should finish its task with:

```text
IMPLEMENTED:
[short summary]

FILES CREATED:
[list]

FILES MODIFIED:
[list]

TESTS RUN:
[list of actual commands/tests]

RESULTS:
[actual measured results only]

ASSUMPTIONS:
[only if necessary]

NEXT STEP:
[what the other member should do]
```

Keep reports short.

---

# 20. Member Ownership Summary

## Member 1

```text
DATA
 ↓
NORMALIZATION
 ↓
BLOCKING
 ↓
CANDIDATE RECALL
 ↓
candidate_pairs.tsv
```

## Member 2

```text
candidate_pairs.tsv
 ↓
PAIR FEATURES
 ↓
MATCHING MODEL
 ↓
THRESHOLD
 ↓
SINGLETON HANDLING
 ↓
matching_results.tsv
```

## Both

```text
VALIDATE
 ↓
ERROR ANALYSIS
 ↓
FINAL SUBMISSION
```

---

# 21. Core Rule

The team must not optimize everything simultaneously.

Use this sequence:

```text
1. Make data handling correct
2. Make candidate recall strong
3. Make pair features useful
4. Train matching model
5. Optimize F0.5 threshold
6. Handle singletons
7. Validate submission
8. Submit
```

Change one major component at a time and measure its effect.
