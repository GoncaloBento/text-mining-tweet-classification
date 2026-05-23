# 📊 Exploratory Data Analysis (EDA) Conclusions
**Nova IMS — Text Mining 2025/2026**

This document summarizes the core analytical findings from the exploratory analysis of the **Finance Tweets Sentiment Corpora** (`notebooks/00_eda.ipynb`). These insights directly inform our feature engineering, class balancing, and preprocessing decisions.

---

## 📈 1. Class Distribution & Imbalance
The training dataset consists of **9,543 tweets**. The target label class distribution is highly imbalanced:

| Sentiment | Label ID | Count | Percentage | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Neutral** | `2` | 6,178 | **64.74%** | News headlines, factual announcements, and press releases. |
| **Bullish** | `1` | 1,923 | **20.15%** | Positive market sentiment, price upgrades, target increases. |
| **Bearish** | `0` | 1,442 | **15.11%** | Negative market sentiment, price downgrades, target decreases. |

> [!WARNING]
> **Key Finding**: Heavy class imbalance. Approximately **65% of the corpus is Neutral**.
> **Action Item**: Simple classification models will over-predict the Neutral class. We must use stratified cross-validation (`StratifiedKFold`), set class weights (e.g., `class_weight='balanced'` in scikit-learn), or explore oversampling techniques (SMOTE) to ensure our models are sensitive to Bearish and Bullish sentiments.

---

## 📏 2. Tweet Length Characteristics
We analyzed the character and word (token) lengths of tweets globally and by class:

* **Global Char Length**: Mean = **85.8 characters** (Std = 35.1), Min = 2, Max = 190.
* **Global Word Count**: Mean = **12.2 tokens** (Std = 4.7), Min = 1, Max = 32.

### Length Breakdown by Class:
| Sentiment Class | Avg Character Length | Avg Word (Token) Count |
| :--- | :--- | :--- |
| **Bearish** | 83.3 chars | 12.0 tokens |
| **Bullish** | 80.4 chars | 11.9 tokens |
| **Neutral** | 88.1 chars | 12.3 tokens |

> [!NOTE]
> **Key Finding**: Tweets in this corpus are highly compact and short. Neutral tweets are slightly longer on average because they often contain full press headlines or URL links. 
> **Action Item**: No massive truncations or sequence length constraints are needed for Transformer models; a maximum sequence length of 64 tokens will comfortably capture all text.

---

## 🔍 3. Tweet Artifacts (URLs, Cashtags, Mentions, Hashtags)
A statistical breakdown reveals that tweets are rich in specialized metadata markers:

* **URLs (46.8% of all tweets)**: Extremely frequent! Average of 0.53 URLs per tweet.
* **Cashtags (15.0% of all tweets)**: Market-specific ticker markers (e.g. `$TSLA`, `$AAPL`, `$SPY`).
* **Hashtags (9.4% of all tweets)**: Content markers (e.g., `#Stock`, `#economy`, `#trading`).
* **Mentions (3.1% of all tweets)**: Very rare (e.g. `@elonmusk`).

### Average Artifact Rates by Class:
| Sentiment Class | URLs Rate | Cashtags Rate | Hashtags Rate | Mentions Rate |
| :--- | :--- | :--- | :--- | :--- |
| **Bearish** | 0.510 | 0.216 | 0.165 | 0.021 |
| **Bullish** | 0.438 | **0.341** | 0.158 | 0.017 |
| **Neutral** | **0.569** | 0.171 | **0.265** | **0.046** |

> [!IMPORTANT]
> **Strategic Insights**:
> 1. **URLs are a Strong Predictor for Neutrality**: 56.9% of Neutral tweets contain URLs (mostly links to articles). Deleting URLs entirely removes this powerful structural feature. Replacing URLs with a standard token like `URL_PLACEHOLDER` preserves this signal for classical classifiers.
> 2. **Cashtags are Highly Bullish**: Bullish tweets contain cashtags at a much higher frequency (**34.1%**) compared to others. The actual stock name (e.g., `$AAPL` vs `$TSLA`) is less critical than the presence of the tag. Replacing ticker symbols with `CASHTAG_PLACEHOLDER` reduces vocabulary size and simplifies learning while preserving this vital feature.

---

## 🛠️ 4. Non-ASCII Characters & Unicode Issues
* **Train Set**: **23.50%** of tweets contain non-ASCII characters.
* **Test Set**: **24.50%** of tweets contain non-ASCII characters.

### Examples of Issues:
1. **Smart Quotes & Apostrophes**: Use of curly quotes (`“underperform”`) and smart apostrophes (`Netflix’s`) which can lead to duplicate vocabulary tokens if not normalized (e.g. `Netflix's` vs `Netflix’s`).
2. **Weird Truncation Marks**: Truncated text endings marked by the replacement character `` or custom ellipses (e.g., `price tar https`).

> [!TIP]
> **Action Item**: Standardizing these characters is essential. The preprocessing pipeline must normalize smart punctuation (curly quotes, dashes, ellipses) to standard ASCII equivalents before tokenization to ensure consistent feature extraction.

---

## 📋 5. Duplicates & Empty Rows
* **Exact duplicates**: 0 in both Train and Test sets.
* **Case-insensitive + Trimmed duplicates**: Only 3 tweets in Train, 0 in Test.
* **Empty/Whitespace rows**: 0.

> [!NOTE]
> **Action Item**: The dataset is exceptionally clean regarding duplication; no special duplicate-dropping or empty-row handling is required in our pipeline.
