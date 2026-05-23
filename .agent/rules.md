# Repository Agent Rules & Instructions
# This file is automatically indexed by AI Agents (Antigravity, Cursor, Claude CLI, and others).

You are an AI assistant pair-programming with the team on the Text Mining course project (Spring 2025/2026).
Follow these guidelines strictly to maintain consistency, reproducibility, and leverage the built-in repository tools.

---

## 🐍 1. Environment & Execution Guidelines
* **Virtual Environment**: Always use the local virtual environment `.venv`.
  * **Windows**: Activate via `.venv\Scripts\activate` or run scripts using `.\.venv\Scripts\python.exe`.
  * **macOS/Linux**: Activate via `source .venv/bin/activate` or run scripts using `./.venv/bin/python`.
* **Dependencies**: Any required libraries are listed in `requirements.txt`. Do not install packages globally; always install them inside the active `.venv`.

---

## 🛠️ 2. Integrated NLP Tools & Pipelines

This repository has three highly optimized, built-in CLI tools designed to be executed by agents to perform EDA, preprocessing, and model experiments. **Always check and use these tools before trying to write your own custom routines.**

### 📊 A. Exploratory Data Analysis Tool (`src/eda.py`)
Run this tool to automatically analyze dataset size, class distributions, character/word length distributions, Twitter metadata artifacts (URLs, hashtags, cashtags, mentions), non-ASCII character rates, and highly informative vocabularies.

* **Run complete analysis and save high-res visual plots under `outputs/eda/`**:
  ```bash
  python src/eda.py --save-plots
  ```
* **Run analysis and output a structured JSON report (perfect for programmatic agent parsing)**:
  ```bash
  python src/eda.py --format json
  ```
* **Analyze custom data files**:
  ```bash
  python src/eda.py --train-path data/train.csv --test-path data/test.csv
  ```

### 🧹 B. Text Preprocessing Pipeline (`src/preprocessing.py`)
Run this tool to clean financial tweets. It includes Unicode/punctuation normalization (for curly smart quotes, em-dashes, ellipses), lowercasing, regex cleaning, stopwords removal, tokenization, stemming, and lemmatization.

* **Pipeline Features & Placeholders**:
  * URLs, user mentions, and stock cashtags are replaced by unified, protected placeholders: `URL_PLACEHOLDER`, `MENTION_PLACEHOLDER`, `CASHTAG_PLACEHOLDER`.
  * These placeholders are protected and **never split, stemmed, or lemmatized** by NLTK.
* **Command-line Interface**:
  ```bash
  # Preprocess a single tweet and output space-joined string:
  python src/preprocessing.py --text "$TSLA is going up! check out https://t.co/example" --url-mode replace --return-str
  ```
* **JSON-Piped Interface (ideal for agent-to-agent workflows)**:
  Provide input JSON via stdin to `python src/preprocessing.py --json-input`.

### 🔬 C. Sentiment Classification Experiment Runner (`src/experiment.py`)
Run this script to automatically train, evaluate, and compare Logistic Regression baselines using three different TF-IDF feature extractions:
1. **Model A: TF-IDF Unigrams Baseline** (`ngram_range=(1,1)`)
2. **Model B: TF-IDF Unigrams + Bigrams Raw** (`ngram_range=(1,2)`)
3. **Model C: TF-IDF Unigrams + Bigrams Optimized** (`ngram_range=(1,2)`, `min_df=2`, `max_features=25000`)

* **Pipeline Automation**:
  * Loads and splits data using `src/train_val_split.py`.
  * Preprocesses text using `src/preprocessing.py`.
  * Evaluates and prints comparative metrics tables.
  * Automatically programmatically generates/updates the formal `notebooks/02_bow_tfidf_classical.ipynb` Jupyter Notebook.
* **Idempotence**:
  The experiment runner logs all results to the rolling project leaderboard `outputs/results.csv` **idempotently**. Running the script multiple times with the same parameters will overwrite and update the existing row rather than appending duplicate rows.
* **Execution**:
  ```bash
  python src/experiment.py
  ```

---

## 🧪 3. Running Automated Unit Tests
Ensure the preprocessing pipeline and the experiment logging system remain fully correct and functional. Run the dedicated unit testing suites automatically before merging or changing code:

* **Run all tests inside the tests directory**:
  ```bash
  python -m unittest discover tests/
  ```
* **Run specific test suites**:
  ```bash
  # Preprocessing pipeline tests
  python -m unittest tests/test_preprocessing.py
  
  # Idempotent experiment logging tests
  python -m unittest tests/test_experiment.py
  ```

---

## 🏆 4. Reproducibility & Course Grading Rules
* **Random Seed**: You MUST enforce a random seed of `42` everywhere (NumPy, PyTorch, Random, Scikit-learn `random_state`).
* **Feature Engineering**: Vectorizers and word embedding models must be fitted on **train data only** to prevent data leakage.
* **No Loose Output Files**: All final deliverables must go into their respective directories:
  * Final test predictions: `outputs/pred_xx.csv` (only two columns: `id` and `label`).
  * leaderboards of model runs: `outputs/results.csv`.
