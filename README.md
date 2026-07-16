# Text Mining 2025-2026 — Project

NLP project for the Text Mining course at NOVA IMS. Goal: classify the market sentiment of finance tweets as **Bearish (0) / Bullish (1) / Neutral (2)**.



## Repository layout

```
.
├── data/                 # train.csv / test.csv
├── notebooks/
│   ├── archive/          # Archived experimental notebooks
│   ├── tm_tests_31.ipynb # Official tested techniques notebook
│   └── tm_final_31.ipynb # Clean final pipeline notebook
├── outputs/
│   ├── eda/              # EDA charts and conclusions
│   ├── embeddings/       # Extracted Transformer and word embeddings
│   ├── results.csv       # Unified rolling leaderboard
│   └── pred_best.csv     # Final predictions deliverable
├── report/               # Report drafts and final LaTeX source
│   └── final_report.tex  # Official Final Report
├── src/                  # Main Python source code and trainers
│   ├── autotune.py
│   ├── config.py
│   ├── eda.py
│   ├── error_analysis.py
│   ├── evaluate.py
│   ├── experiment.py
│   ├── features.py
│   ├── preprocessing.py
│   ├── word_embeddings.py
│   └── *_trainer.py      # Various transformer training scripts
└── README.md
```

## Setup (first time)

```bash
# 1. clone
git clone https://github.com/RedThreat/TextMining-Corpora.git
cd TextMining-Corpora

# 2. create a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. install dependencies
pip install -r requirements.txt

# 4. drop train.csv and test.csv into data/
```

## Daily workflow

1. Pull `main`, branch off (`git checkout -b feat/<short-name>`).
2. One owner per notebook — rebase before merging.
3. Append every model run to `outputs/results.csv` via the helper in `src/evaluate.py`.
4. Open a PR; another team member reviews; squash-merge.

## Reproducibility rules

- Random seed `42` everywhere (numpy, random, torch, sklearn `random_state`).
- Vectorisers and embedding models fitted on **train only**.
- Final pipeline must run top-to-bottom on a clean kernel before submission.


## 📋 Course Project Guidelines

These guidelines are summarized from the official [Project Guidelines - Text Mining 2025-2026 v2.pdf]

### 🎯 Objective & Summary
* **Goal:** Develop an NLP classification model capable of predicting market sentiment from financial tweets as:
  * **Bearish (0)**
  * **Bullish (1)**
  * **Neutral (2)**
* **Tech Stack:** Python 3, using libraries such as `NLTK`, `Scikit-Learn`, `Hugging Face`, and `LangChain`.


