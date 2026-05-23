---
name: text_preprocessing
description: Cleans, tokenizes, normalizes, and lemmatizes raw financial tweets using protected placeholders for mentions, URLs, and cashtags.
---

# Text Preprocessing & Tokenization Skill

This skill provides execution interfaces, functional specifications, and validation instructions for cleaning raw financial text data using our custom NLP pipeline in [preprocessing.py](file:///c:/Users/filip/TextMining-Corpora/src/preprocessing.py).

## 📋 Skill Prerequisites
- Local virtual environment `.venv` active.
- NLTK downloaded resources (Stopwords, WordNet Lemmatizer, etc.).

## 🧹 Pipeline Architecture
Our custom pipeline implements advanced NLP normalizations:
1. **Unicode & Punctuation Normalization**: Sanitizes smart quotes (`“`/`”`), em-dashes (`—`), ellipses (`…`), and standardizes spacing.
2. **Entity Protection**: Replaces URLs, user mentions, and stock tickers (cashtags) with protected tokens to prevent stemming/lemmatization:
   - `https://t.co/xyz` $\rightarrow$ `URL_PLACEHOLDER`
   - `@username` $\rightarrow$ `MENTION_PLACEHOLDER`
   - `$TSLA` or `$AAPL` $\rightarrow$ `CASHTAG_PLACEHOLDER`
3. **Regex Sanitization**: Discards leftover non-alphanumeric junk characters while protecting our placeholder tokens.
4. **Tokenization & Stopwords**: Tokenizes text, handles negation boundary markings, and removes sparse/meaningless English stop words.
5. **Lemmatization**: Applies NLTK WordNet Lemmatizer to group word variants (e.g., `running` $\rightarrow$ `run`, `stocks` $\rightarrow$ `stock`).

---

## 🛠️ Execution & Validation Instructions

### 1. Interactive Console Test
Verify preprocessing results on a raw financial tweet and print the cleaned token string:
```bash
python src/preprocessing.py --text "Check \$AAPL out! High momentum at @website https://t.co/test" --return-str
```
*Expected Cleaned Output:*
`check CASHTAG_PLACEHOLDER out high momentum at MENTION_PLACEHOLDER URL_PLACEHOLDER`

### 2. Output as Token List
Test parsing the cleaned text as a structured python token list (perfect for word embedding pipelines):
```bash
python src/preprocessing.py --text "We are buying shares of \$TSLA tomorrow"
```

### 3. Programmatic Stdin Stream
You can pipe multiple raw texts via stdin using JSON formatting. This is the fastest way to run bulk text mining tasks across tools:
```bash
echo '{"text": "Extremely bullish on \$AMZN earnings!"}' | python src/preprocessing.py --json-input
```

---

## 🧪 Pipeline Unit Testing
Always run the test suite to verify that any adjustments to regex rules or placeholder boundaries do not break normalizations:
```bash
python -m unittest tests/test_preprocessing.py
```
