# 1. Exploratory Data Analysis (EDA)
**Nova IMS — Text Mining 2025/2026**

This section presents a rigorous, detailed exploratory data analysis of the **Finance Tweets Sentiment Corpora**. Our primary goal is to examine the structural, semantic, and metadata properties of the corpus to inform subsequent feature engineering, preprocessing strategies, and classification model selections.

---

## 📊 1.1. Dataset Scale & Class Distribution

The training dataset consists of **9,543 tweets**. The target labels categorize sentiment as **Bearish (0)**, **Bullish (1)**, or **Neutral (2)**. Our analysis reveals a severe class imbalance:

| Sentiment Class | Label ID | Count | Percentage | Description / Content Characteristics |
| :--- | :---: | :---: | :---: | :--- |
| **Neutral** | `2` | 6,178 | **64.74%** | Factual journalism, press releases, corporate results. |
| **Bullish** | `1` | 1,923 | **20.15%** | Positive analyst upgrades, target revisions, stock jumps. |
| **Bearish** | `0` | 1,442 | **15.11%** | Downgrades, market crashes, missed forecasts, warnings. |
| **Total** | — | **9,543** | **100.00%** | **Entire training partition** |

> [!WARNING]
> **Heavy Class Imbalance (64.7% Majority Class)**:
> Since approximately **65% of the tweets are labeled Neutral**, standard machine learning classifiers are highly susceptible to majority class bias, yielding deceptively high accuracies but poor minority class sensitivity. To resolve this:
> 1. We must enforce stratified validation splits (`StratifiedKFold`) to preserve label ratios.
> 2. We must use balanced class weights (e.g. `class_weight='balanced'`) to penalize minority misclassifications more severely.
> 3. We must evaluate models using **Macro F1-Score** rather than global accuracy.

---

## 📏 1.2. Tweet Length Characteristics

We profiled the character and word length distributions of the tweets across classes to assess sequential density:

* **Global Character Length**: Mean = **85.8 characters** (Std = 35.1), Min = 2, Max = 190.
* **Global Word (Token) Count**: Mean = **12.2 tokens** (Std = 4.7), Min = 1, Max = 32.

### Average Length Breakdown by Sentiment Class:
| Sentiment Class | Avg Character Length | Avg Word (Token) Count |
| :--- | :---: | :---: |
| **Bearish** | 83.3 characters | 12.0 tokens |
| **Bullish** | 80.4 characters | 11.9 tokens |
| **Neutral** | 88.1 characters | 12.3 tokens |

*Core Observation*: The texts are highly compact and compressed, with a maximum token length of 32 words. Neutral tweets are slightly longer on average because they contain formal news headlines and URL links.
*Downstream Action*: Because sequences are extremely short, we do not require heavy padding or truncation operations. A maximum token sequence length of 64 is more than sufficient for downstream Transformer models, avoiding unnecessary compute waste.

---

## 🔍 1.3. Specialized Twitter Metadata Artifacts

Financial tweets are heavily structured around metadata markers: **URLs** (links to news), **Cashtags** (stock tickers like `$AAPL`), **Hashtags** (topics), and **Mentions** (user interactions). 

Our statistical analysis reveals that the frequencies of these artifacts are **highly predictive** of specific sentiment classes:

| Sentiment Class | URL Rate (per tweet) | Cashtag Rate (per tweet) | Hashtag Rate (per tweet) | Mention Rate (per tweet) |
| :--- | :---: | :---: | :---: | :---: |
| **Bearish (0)** | 0.510 | 0.216 | 0.165 | 0.021 |
| **Bullish (1)** | 0.438 | **0.341** | 0.158 | 0.017 |
| **Neutral (2)** | **0.569** | 0.171 | **0.265** | **0.046** |
| **Global Average** | **0.530** | **0.201** | **0.218** | **0.038** |

### 💡 Strategic Artifact Insights:
1. **URLs strongly signal Neutrality**: Over **56.9%** of Neutral tweets contain URLs (mostly linking to external earnings articles). Deleting URLs entirely deletes a vital structural signal. Replacing URLs with a unified `URL_PLACEHOLDER` token preserves this layout context for models.
2. **Cashtags are highly Bullish indicators**: Bullish tweets contain cashtags at a dramatically higher rate (**34.1%**) compared to other classes. The presence of a cashtag (e.g. `$TSLA`) is far more informative than the actual stock name. Replacing stock tickers with a unified `CASHTAG_PLACEHOLDER` reduces vocabulary dimensionality while preserving this vital semantic marker.

---

## 🛠️ 1.4. Non-ASCII Characters & Unicode Anomalies

A significant portion of our dataset contains non-ASCII punctuation and characters:
* **Train Set**: **23.50%** of tweets contain non-ASCII characters.
* **Test Set**: **24.50%** of tweets contain non-ASCII characters.

### Key Categories of Unicode Anomalies:
1. **Punctuation Variants**: Frequent use of curly quotes (`“` / `”`), smart apostrophes (`’`), and long en-dashes (`—` / `–`). If left untreated, these create duplicate token entries (e.g. `Amazon's` vs `Amazon’s`), diluting feature density.
2. **Ellipses and Truncations**: Twitter posts are often truncated mid-sentence, generating custom ellipses (`…`) or the replacement character `\uFFFD`. These must be standardized to ASCII `...` to avoid tokenizing them as rare unknown words.

*Strategic Action*: The preprocessing pipeline must implement a robust Unicode/punctuation normalizer that cleans smart punctuation and ellipses to standard ASCII counterparts before tokenization.

---

## 📋 1.5. Duplicates & Empty Rows

* **Exact duplicates**: 0 in both Train and Test sets.
* **Case-insensitive + Trimmed duplicates**: Only 3 tweets in the Train set, 0 in the Test set.
* **Empty/Whitespace rows**: 0.

*Conclusion*: The dataset is exceptionally clean regarding duplicate rows, requiring no custom row-dropping steps in our pipeline.

---

## 📈 1.6. Exploratory Data Analysis Charts

The visual charts generated during our EDA (`outputs/eda/`) illustrate these structural properties:
* **Class Distribution Bar Chart**: [class_distribution.png](file:///c:/Users/filip/TextMining-Corpora/outputs/eda/class_distribution.png)
* **Metadata Rates by Class**: [artefact_rates_by_class.png](file:///c:/Users/filip/TextMining-Corpora/outputs/eda/artefact_rates_by_class.png)
* **Token Length Distributions**: [length_distributions.png](file:///c:/Users/filip/TextMining-Corpora/outputs/eda/length_distributions.png)
* **Informative Words Wordcloud**: [wordclouds_by_class.png](file:///c:/Users/filip/TextMining-Corpora/outputs/eda/wordclouds_by_class.png)
