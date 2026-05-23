---
name: unit_testing_validation
description: Executes the automated unit testing suite across preprocessing and modeling pipelines to detect regressions and mathematically verify idempotence.
---

# Unit Testing & Pipeline Validation Skill

This skill documents instructions, test structures, command interfaces, and validation criteria to verify the logical correctness of the codebase.

## 📋 Skill Prerequisites
- Local virtual environment `.venv` active.
- Core test targets: `tests/test_preprocessing.py` and `tests/test_experiment.py`.

---

## 🛠️ Testing Commands

### 1. Run Complete Test Suite
To automatically discover and execute all unit tests in the repository:
```bash
# Windows
.\.venv\Scripts\python.exe -m unittest discover tests/

# macOS/Linux
./.venv/bin/python -m unittest discover tests/
```

### 2. Run Preprocessing Pipeline Tests
Test that Unicode normalizations, entities placeholders, tokenizations, and lemma mappings behave correctly:
```bash
python -m unittest tests/test_preprocessing.py
```

### 3. Run Experiment Pipeline & Idempotency Tests
Verify that Logistic Regression model fitting works on mock datasets, generalized model training executes cleanly, and `log_model_run` behaves idempotently:
```bash
python -m unittest tests/test_experiment.py
```

---

## 🔬 Test Assertions & Validation Criteria

- **Idempotency Assertions**:
  The suite verifies that duplicate runs with identical parameters rewrite [results.csv](file:///c:/Users/filip/TextMining-Corpora/outputs/results.csv) in-place without increasing the row count of the leaderboard, while runs with differing parameters correctly append new rows.
- **Data Dimension Assertions**:
  The suite verifies that TF-IDF vectors train Logistic Regression models successfully and return valid float metrics (accuracy, F1-score) greater than or equal to 0.0.
- **Regression Checks**:
  Always execute the complete test suite before checking in code modifications or pushing to remote branches.
