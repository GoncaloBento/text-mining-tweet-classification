---
name: eda_data_profiling
description: Executes exploratory data analysis pipelines on financial tweet sentiment datasets to profile metrics, classes, length distributions, and metadata highlights.
---

# Exploratory Data Analysis & Data Profiling Skill

This skill provides step-by-step instructions and command interfaces for running, analyzing, and plotting dataset profiling metrics for our sentiment classification task.

## 📋 Skill Prerequisites
- Local virtual environment `.venv` active.
- Core dependencies (`pandas`, `matplotlib`, `seaborn`) installed.

## 🛠️ Execution Instructions

Use the built-in [eda.py](file:///c:/Users/filip/TextMining-Corpora/src/eda.py) tool to profile any text mining dataset.

### 1. General Profile
Run a standard analytical run that prints class imbalances, vocabulary stats, and character rates to console:
```bash
# Windows
.\.venv\Scripts\python.exe src/eda.py

# macOS/Linux
./.venv/bin/python src/eda.py
```

### 2. Generate and Save Visual Plots
Add the `--save-plots` flag to generate and save statistical plots (distribution graphs, boxplots, class charts) under `outputs/eda/`:
```bash
python src/eda.py --save-plots
```
*Outputs generated:*
- `outputs/eda/class_distribution.png`: Imbalance verification chart.
- `outputs/eda/char_length_distribution.png`: Box and kernel-density plot of token sizes.
- `outputs/eda/top_n_ngrams.png`: Chart showing top vocabulary unigrams/bigrams.

### 3. Programmatic JSON Serialization
Run with `--format json` to get structured data back. This is perfect for parsing metrics programmatically into other scripts/agents:
```bash
python src/eda.py --format json
```

### 4. Direct Custom Files Profiling
If a new split or test data needs analysis, pass paths explicitly:
```bash
python src/eda.py --train-path data/custom_train.csv --test-path data/custom_test.csv
```

---

## 📈 Key Insights to Extract
- **Imbalance Check**: Note the heavily dominant class (usually `Neutral` in financial tweets) to configure class weighting.
- **Outlier Detection**: Spot extreme character lengths or spam links that need cleaning in preprocessing.
- **Word Overlaps**: Analyze high-frequency sentiment unigrams to assess potential features.
