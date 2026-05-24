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
Run this script to automatically train, evaluate, and compare traditional ML configurations (KNN, Logistic Regression, Multinomial NB, MLP, RF, XGB) on the Optimized TF-IDF representation.

* **Pipeline Automation**:
  * Preprocesses text using `src/preprocessing.py`.
  * Logs all results to `outputs/results.csv` **idempotently**.
* **Execution**:
  ```bash
  python src/experiment.py
  ```

### ⚙️ D. Automated Hyperparameter Tuner & Predictor (`src/autotune.py`)
Run this script to programmatically locate the champion classical classifier based on Macro F1, refit it on 100% of the training dataset to maximize knowledge utilization, and generate optimized final submissions in `outputs/pred_best.csv`.

* **Execution**:
  ```bash
  python src/autotune.py
  ```

### 📊 E. Deep Sentiment Error Analysis Pipeline (`src/error_analysis.py`)
Run this script to profile the structural failure modes of the champion classical baseline. It computes and saves the confusion matrix heatmap (`outputs/confusion_matrix.png`) and extracts the top 20 most confident/severe misclassifications per class to `outputs/misclassified_report.txt` (for humans) and `outputs/misclassified_analysis.json` (for agents).

* **Execution**:
  ```bash
  python src/error_analysis.py
  ```

### 🧬 F. Word Embeddings Comparison Pipeline (`src/word_embeddings.py`)
Run this script to automatically compare custom Word2Vec embeddings trained on our corpus against Stanford's pre-trained `glove-twitter-100` model. It computes unique and token Out-of-Vocabulary (OOV) statistics on the validation fold, vectorizes tweets using Mean Pooling, and trains baseline Logistic Regression classifiers, registering the outcomes in our leaderboard.

* **Execution**:
  ```bash
  python src/word_embeddings.py
  ```

### 📊 G. Classical Feature Vectorization (`src/features.py`)
Run this script to automatically extract and save unigram CountVectorizer (BoW) and TfidfVectorizer features. It runs the stratified split, applies full preprocessors, fits the vectorizers **strictly on training data only** to prevent data leakage, and exports the resulting sparse matrices in compressed `.npz` format to the `outputs/` folder.

* **Execution**:
  ```bash
  python src/features.py
  ```

### 💨 H. Baseline Smoke Test & Unified Evaluation (`src/baseline_smoke_test.py` & `src/evaluate.py`)
* **`src/baseline_smoke_test.py`**: Runs a simple majority-class smoke test baseline, producing predictions saved under `outputs/Baseline_Smoke_Test/`.
  ```bash
  python src/baseline_smoke_test.py
  ```
* **`src/evaluate.py`**: Central evaluation framework. Computes accuracy, macro recall, precision, and F1 metrics, displaying confusion matrices and classification reports. It provides a robust, idempotent leaderboard logging engine that supports multiple teammate signatures cleanly while resolving potential column-shifting bugs.

---

## 🧪 3. Running Automated Unit Tests
Ensure the preprocessing pipeline, experiment loggers, and vector pooling systems remain fully correct and functional. Run the dedicated unit testing suites automatically before merging or changing code:

* **Run all tests inside the tests directory**:
  ```bash
  python -m unittest discover tests/
  ```
* **Run specific test suites**:
  ```bash
  # Preprocessing pipeline tests
  python -m unittest tests/test_preprocessing.py
  
  # Sparse matrix feature vectorization tests
  python -m unittest tests/test_features.py
  
  # Idempotent experiment logging tests
  python -m unittest tests/test_experiment.py

  # Word embeddings pooling and OOV rates tests
  python -m unittest tests/test_word_embeddings.py
  ```


---

## 🏆 4. Reproducibility & Course Grading Rules
* **Random Seed**: You MUST enforce a random seed of `42` everywhere (NumPy, PyTorch, Random, Scikit-learn `random_state`).
* **Feature Engineering**: Vectorizers and word embedding models must be fitted on **train data only** to prevent data leakage.
* **No Loose Output Files**: All final deliverables must go into their respective directories:
  * Final test predictions: `outputs/pred_xx.csv` (only two columns: `id` and `label`).
  * leaderboards of model runs: `outputs/results.csv`.
