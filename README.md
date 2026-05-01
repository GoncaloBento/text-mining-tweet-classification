# Text Mining 2025-2026 — Project

NLP project for the Text Mining course at NOVA IMS. Goal: classify the market sentiment of finance tweets as **Bearish (0) / Bullish (1) / Neutral (2)**.

## Quick links

- Notion kanban — `Nova IMS Projects → Text Mining`
- Group: 4 members
- **Internal deadline:** 2026-06-05 · **Hard deadline:** 2026-06-15

## Repository layout

```
.
├── data/                 # train.csv / test.csv (not in git — drop them here)
├── notebooks/
│   ├── 00_eda.ipynb
│   ├── 01_preprocessing.ipynb
│   ├── 02_bow_tfidf_classical.ipynb
│   ├── 03_word_embeddings.ipynb
│   ├── 04_transformers.ipynb
│   ├── 05_decoder_extra.ipynb
│   ├── 06_agent.ipynb
│   ├── tm_tests_xx.ipynb
│   └── tm_final_xx.ipynb
├── src/
│   ├── preprocessing.py
│   ├── features.py
│   ├── evaluate.py
│   └── agent.py
├── outputs/
│   ├── eda/              # EDA charts (committed)
│   ├── results.csv       # rolling leaderboard of every model run
│   └── pred_xx.csv       # final deliverable
├── report/
│   └── report_xx.pdf
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
