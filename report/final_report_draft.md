# Text Mining Project Final Report
**Nova IMS — Text Mining 2025/2026**

## 1. Data Exploration

Our exploratory data analysis of the Finance Tweets Sentiment Corpora reveals a highly compact but heavily imbalanced dataset. 
The training dataset consists of **9,543 tweets** categorized into Bearish (0), Bullish (1), and Neutral (2) sentiments.

### Class Imbalance
* **Neutral**: 6,178 (64.74%)
* **Bullish**: 1,923 (20.15%)
* **Bearish**: 1,442 (15.11%)

This severe majority-class bias requires strict stratified splitting and minority-class compensatory techniques (e.g., balanced class weights) to avoid deceptive accuracy metrics.

### Text Characteristics and Metadata
Tweets are highly compressed, averaging 85.8 characters and 12.2 tokens. Financial texts are uniquely structured:
* **URLs**: Appear in 53.0% of tweets, strongly indicating Neutral factual journalism (56.9% rate).
* **Cashtags**: Stock tickers (e.g., `$AAPL`) appear in 20.1% of tweets and are strong Bullish indicators (34.1% rate).
* **Non-ASCII Characters**: Over 23% of tweets contain smart punctuation, requiring robust normalization.

---

## 2. Corpus Split and Data Preprocessing

To preserve label ratios, we partitioned our dataset using a strict **80/20 stratified train-validation split** (7,634 train; 1,909 validation).

We designed a robust 7-step preprocessing pipeline specifically tailored for financial microblogs:
1. **Unicode & Punctuation Normalization**: Converting smart quotes and en-dashes to standard ASCII.
2. **Lowercasing**: Collapsing identical terms.
3. **Regex Metadata Cleaning**: Replacing sparse, unique URLs and user handles with protected placeholders (`URL_PLACEHOLDER`, `MENTION_PLACEHOLDER`).
4. **NLTK TweetTokenizer**: Preserving hashtags and financial symbols.
5. **Custom Stopwords Removal**: Removing standard stopwords plus financial noise terms (e.g., "rt", "http", "share").
6. **WordNet Lemmatization**: Mapping words to roots without the destructive truncation of standard stemmers.

---

## 3. Feature Engineering

We transitioned from discrete Bag-of-Words to continuous dense vector spaces, exploring multiple representations.

### Classical Vectorization
We implemented **CountVectorizer (BoW)** and **TfidfVectorizer** for unigrams. BoW slightly outperformed TF-IDF on Macro F1 (0.6990 vs. 0.6957) because in ultra-short financial tweets, raw token presence is a stronger sentiment indicator than inverse-frequency dampened weights. 

### Word Embeddings
We compared a **Custom Word2Vec model** (trained from scratch) against a **pre-trained GloVe-Twitter-100 model**.
* **OOV Rate**: Custom Word2Vec had 0% OOV for localized tokens and placeholders, whereas GloVe missed 33% of unique validation terms (failing on cashtags and custom jargon).
* **Pooling Strategies**: We explored Mean Vector Pooling vs. TF-IDF Weighted Average Pooling. Counter-intuitively, Mean Pooling outperformed TF-IDF weighting, as the latter over-amplified rare, noisy slang terms.
* **The Pooling Bottleneck**: Both continuous models struggled against the optimized high-dimensional TF-IDF unigram+bigram baseline because averaging word vectors destroys word-order and negation boundaries.

**[EXTRA WORK - FEATURE ENGINEERING]**: 
To resolve static pooling limitations, we expanded our architecture to use **Dynamic Contextual Transformer Embeddings**, leveraging the self-attention mechanisms of RoBERTa and FinBERT to generate context-aware representations that natively handle negation and complex clauses.

---

## 4. Classification Models

We systematically benchmarked 16 classical machine learning configurations across six algorithms (KNN, Logistic Regression, MNB, MLP, Random Forest, XGBoost) using Optimized TF-IDF features.

### Prioritized Analysis: Handling Class Imbalance
Based on the severe 65% Neutral class imbalance, we ran an exhaustive experiment grid comparing four balancing strategies: **`class_weight='balanced'`**, **`oversample`**, **`undersample`**, and **`none`** across model families.

**Key Findings on Imbalance Strategies (Macro F1):**
* **Logistic Regression (LR)**: `class_weight` (0.7113) > `oversample` (0.7085) > `undersample` (0.6652) > `none` (0.6417).
* **Multi-Layer Perceptron (MLP)**: `oversample` (0.6884) > `none` (0.6830).
* **Random Forest (RF)**: `class_weight` (0.6589) > `oversample` (0.6326) > `none` (0.3313).
* **Winner**: **Logistic Regression + `class_weight='balanced'`** emerged as the absolute champion (F1 Macro = 0.7113). By mathematically scaling the loss function inversely to class frequencies, LR maximized recall on the minority Bearish (63.54%) and Bullish (67.79%) classes without losing Neutral accuracy. Over-sampling and Under-sampling produced competitive but slightly inferior bounds due to noise replication and data loss, respectively.

**[EXTRA WORK - CLASSIFICATION MODELS]**:
Going beyond classical ML, we implemented and fine-tuned two advanced contextual Transformer models:
1. **Twitter-RoBERTa-base-sentiment**: Pre-trained on 58M tweets, natively handling informal syntax, slang, and emoticons.
2. **FinBERT (ProsusAI)**: Pre-trained on formal financial texts, excelling in precise financial jargon extraction.
Both models represent a leap in architecture, actively overcoming the limitations of static word embeddings. We explicitly handled their mismatched label schemas by safely re-initializing their classification heads while freezing the backbone weights.

---

## 5. Evaluation and Results

Our champion classical baseline (**Logistic Regression L2, class_weight='balanced', Optimized TF-IDF**) achieved:
* **Validation Accuracy**: 0.7810
* **Macro Precision**: 0.7041
* **Macro Recall**: 0.7201
* **Macro F1-Score**: 0.7113

### Error Analysis & Failure Modes
Despite high performance, confusion matrix analysis reveals distinct limitations of linear BoW models:
1. **Contextual Negation Blindness**: Failing to resolve sequential dependencies (e.g., `"not to like"`, `"fail to halt"`), leading to polarity inversions.
2. **Factual Reporting vs. Sentiment**: The model struggles with boundary ambiguity, aggressively flagging factual positive corporate news (often labeled Neutral) as Bullish.
3. **Clause Overriding**: In complex tweets (e.g., *"profit rise fails to salvage"*), the sum of positive token weights overrides single critical negative tokens.
These failures definitively justify our advanced transition into context-aware deep learning models (FinBERT/RoBERTa) capable of gating semantic flow.

---

## 6. Extra Challenge: Agentic AI Workflow

To meet the Extra Challenge criteria, our development process was heavily orchestrated by an **Agentic AI Workflow**. 
The AI assistant performed non-trivial decision-making and codebase engineering, including:
* **Developing a Smart Dispatcher (`autotune.py`)**: The agent dynamically refactored our tuning pipeline into an intelligent router. It parses leaderboard outputs (`outputs/results.csv`) and seamlessly dispatches execution. 
* **Automating Deep Learning Evaluation**: If the champion model is a Transformer (e.g., DistilBERT), it automatically imports the necessary trainer module and runs the PyTorch loops, rather than failing or restricting evaluation to classical algorithms.
* **Classical ML Refitting**: If the champion is classical, it safely refits on the full 100% training split and generates the final test prediction CSVs (`outputs/pred_best.csv`).
* **Architectural Coherence**: The agent proactively identified inconsistencies between our class-imbalance Jupyter Notebooks and our `.csv` leaderboards, programmatically extracting and logging the missing data to ensure the pipeline correctly recognized the true baseline champion.
