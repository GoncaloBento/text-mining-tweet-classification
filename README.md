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

## 🤖 AI Agent Integration

This repository is fully optimized for AI coding assistants (Antigravity, Cursor, Claude CLI, etc.) using a dedicated repository-scoped agent configuration under the [.agent/](file:///c:/Users/filip/TextMining-Corpora/.agent/) directory:

* **📁 [.agent/rules.md](file:///c:/Users/filip/TextMining-Corpora/.agent/rules.md)**: Universal rules and guidelines (reproducibility, testing, built-in tools usage) that all agents should read upon opening this workspace.
* **📁 [.agent/skills/](file:///c:/Users/filip/TextMining-Corpora/.agent/skills/)**: Directory for custom modular tools, workflows, and skill definitions that travel with this repository.
* **📁 [.agent/agents/](file:///c:/Users/filip/TextMining-Corpora/.agent/agents/)**: Directory for custom agent prompt templates, role definitions, and system configurations.

Incoming coding assistants can execute the built-in exploratory data analysis (`src/eda.py`) and text preprocessing (`src/preprocessing.py`) tools by following the exact command structures documented inside [.agent/rules.md](file:///c:/Users/filip/TextMining-Corpora/.agent/rules.md).

---

## 📋 Course Project Guidelines

These guidelines are summarized from the official [Project Guidelines - Text Mining 2025-2026 v2.pdf]

### 🎯 Objective & Summary
* **Goal:** Develop an NLP classification model capable of predicting market sentiment from financial tweets as:
  * **Bearish (0)**
  * **Bullish (1)**
  * **Neutral (2)**
* **Tech Stack:** Python 3, using libraries such as `NLTK`, `Scikit-Learn`, `Hugging Face`, and `LangChain`.
* **Group Size:** 1 to 4 students.

### 📅 Deliverables & Deadlines
* **Submission Deadline:** **Midnight, 15th of June 2026** (submitted via Moodle in a folder named `group_xx`).
* **Required Files:**
  1. `tm_tests_xx.ipynb`: Notebook containing all experimented techniques and their evaluation.
  2. `tm_final_xx.ipynb`: A clean, ready-to-run notebook containing only the final solution (single pipeline with a single classification model).
  3. `pred_xx.csv`: Test set predictions (only two columns: the ID of the test set and predicted labels).
  4. `report_xx.pdf`: A PDF report documenting the work (maximum **15 pages**).

### 📊 Evaluation & Solution Requirements
Your project grade (out of 20 points) is determined by:
1. **Data Exploration (2.00 pts):** Analyze the corpora, provide visual charts (bar charts, word clouds, etc.), and draw conclusions.
2. **Corpus Split (0.50 pts):** Split the training set into train/validation (or K-Fold cross validation) to evaluate performance.
3. **Data Preprocessing (3.00 pts):** Implement at least **four (4)** preprocessing techniques taught in class (regular expressions, stop words, lemmatization, stemming, etc.).
4. **Feature Engineering (5.50 pts):** Implement and experiment with at least one variation of each of: `BoW`, `word2vec`, and `Transformer Encoder`.
5. **Classification Models (4.50 pts):** Implement and test at least two variations of:
   * **Traditional ML** (KNN, MLP, Logistic Regression, Random Forest, XGBoost, etc.)
   * **Transformer Encoders**
6. **Evaluation and Analysis (1.50 pts):** Evaluate models using Recall, Precision, Accuracy, and F1-Score, and analyze their real-world meaning.

#### ✨ Extra Work (Max +2.00 pts):
* **Extra Feature Engineering (+0.50 pts):** Each extra Transformer Encoder method applied (max 2 extra methods).
* **Classification Models (+1.00 pt):** Correctly using a **Decoder model** (LLM) for classification.

### 🏆 Extra Challenges (Grades capped at 20)
* **Extra Challenge 1 (Performance Leaderboard):**
  * 🥇 Best model: **+1.00 point**
  * 🥈 2nd best: **+0.50 point**
  * 🥉 3rd best: **+0.25 point**
* **Extra Challenge 2 (Agentic AI workflow - up to +1.50 points):**
  * Correctly design and implement an **agentic AI-based workflow** orchestrating the classification pipeline using tools or multiple models.
  * Must feature a **conversational interface** that performs a non-trivial coordination/decision task (e.g., choosing between models, comparing classifier outputs, routing tweets, or automating evaluation).

### ⚠️ Penalties & Plagiarism
* **Page Limit Penalty:** **0.5-point penalty** per page exceeding the 15-page report limit.
* **Late Submission:** **1.0-point penalty** per half-day late.
* **Delivery Guide Non-Compliance:** Up to a **1.0-point penalty**.
* **Academic Integrity:** Randomly selected students may be called for an oral defense of their code.

