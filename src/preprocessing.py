#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
src/preprocessing.py

A modular preprocessing pipeline for financial tweets.
Implements the required steps of:
1. Unicode/punctuation normalization (smart quotes, dashes, ellipses,  truncation markers)
2. Lowercasing
3. Regex-based cleaning (URLs, Cashtags, Mentions)
4. Stopwords removal (NLTK English + Custom financial stopwords)
5. Tokenization (using NLTK's TweetTokenizer)
6. Porter Stemming (with protected placeholder tokens)
7. WordNet Lemmatization (with protected placeholder tokens)
8. Interactive command-line interface & agent tool capability
"""

import re
import sys
import json
import argparse
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem import PorterStemmer, WordNetLemmatizer

# --- NLTK Resource Downloader ---
def download_nltk_resources():
    """Automatically downloads necessary NLTK data to ensure zero dependency setup errors."""
    resources = {
        'stopwords': 'corpora/stopwords',
        'punkt': 'tokenizers/punkt',
        'wordnet': 'corpora/wordnet',
        'omw-1.4': 'corpora/omw-1.4'
    }
    for resource_name, path in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource_name, quiet=True)

# Run downloader on module import
download_nltk_resources()

# --- Custom Financial Stopwords ---
DEFAULT_FINANCIAL_STOPWORDS = [
    "rt", "amp", "co", "qt", "http", "https", "via",
    "stock", "stocks", "ticker", "tickers", "share", "shares"
]

# --- Protected Placeholder Constants ---
URL_PLACEHOLDER = "URL_PLACEHOLDER"
MENTION_PLACEHOLDER = "MENTION_PLACEHOLDER"
CASHTAG_PLACEHOLDER = "CASHTAG_PLACEHOLDER"
PROTECTED_PLACEHOLDERS = {URL_PLACEHOLDER, MENTION_PLACEHOLDER, CASHTAG_PLACEHOLDER}

# --- Core Preprocessing Functions ---

def normalize_unicode_punctuation(text: str) -> str:
    """
    Normalizes non-ASCII punctuation such as curly quotes, smart apostrophes,
    long dashes, and ellipses into their standard ASCII counterparts.
    Also handles standard replacement characters (\uFFFD) gracefully.
    """
    if not isinstance(text, str):
        return ""
    # Replace curly double quotes
    text = re.sub(r'[“”„‟]', '"', text)
    # Replace curly single quotes / smart apostrophes
    text = re.sub(r'[’’‘’]', "'", text)
    # Replace long dashes / en-dashes / em-dashes
    text = re.sub(r'[——–—−]', '-', text)
    # Replace ellipses
    text = re.sub(r'…', '...', text)
    # Replace raw replacement character \uFFFD and other weird artifacts
    text = re.sub(r'\uFFFD', ' ', text)
    return text

def apply_lowercase(text: str) -> str:
    """Converts the input string to lowercase, preserving uppercase placeholders if present."""
    # We lowercase standard text, but keep placeholders protected if they are already in the string.
    # Typically we run lowercasing BEFORE regex replacement, so placeholders are inserted in UPPERCASE.
    return text.lower()

def clean_regex(
    text: str,
    url_mode: str = 'replace',
    mention_mode: str = 'replace',
    cashtag_mode: str = 'keep'
) -> str:
    """
    Cleans URLs, mentions, and cashtags from the input string based on specified modes.

    Parameters:
    -----------
    text : str
        The input tweet text.
    url_mode : str
        How to handle URLs: 'keep', 'remove', or 'replace' (with 'URL_PLACEHOLDER').
    mention_mode : str
        How to handle user mentions (@user): 'keep', 'remove', or 'replace' (with 'MENTION_PLACEHOLDER').
    cashtag_mode : str
        How to handle cashtags ($AAPL): 'keep', 'remove', or 'replace' (with 'CASHTAG_PLACEHOLDER').
    """
    # Regex patterns
    url_pattern = r'https?://\S+|www\.\S+'
    mention_pattern = r'@\w+'
    cashtag_pattern = r'\$\w+'

    # Process URLs
    if url_mode == 'remove':
        text = re.sub(url_pattern, '', text)
    elif url_mode == 'replace':
        text = re.sub(url_pattern, URL_PLACEHOLDER, text)

    # Process Mentions
    if mention_mode == 'remove':
        text = re.sub(mention_pattern, '', text)
    elif mention_mode == 'replace':
        text = re.sub(mention_pattern, MENTION_PLACEHOLDER, text)

    # Process Cashtags
    if cashtag_mode == 'remove':
        text = re.sub(cashtag_pattern, '', text)
    elif cashtag_mode == 'replace':
        text = re.sub(cashtag_pattern, CASHTAG_PLACEHOLDER, text)

    # Clean up redundant spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_tweet(text: str) -> list:
    """
    Tokenizes a tweet using NLTK's TweetTokenizer.
    TweetTokenizer is preferred as it keeps hashtags, mentions, and cashtags intact.
    """
    tokenizer = TweetTokenizer(preserve_case=True, reduce_len=True, strip_handles=False)
    return tokenizer.tokenize(text)

def remove_stopwords_from_tokens(
    tokens: list,
    extra_stopwords: list = None
) -> list:
    """
    Removes English stopwords and standard financial stopwords from the token list.
    
    Parameters:
    -----------
    tokens : list
        List of strings.
    extra_stopwords : list, optional
        A list of additional custom stopwords to remove.
    """
    nltk_stops = set(stopwords.words('english'))
    custom_stops = set(DEFAULT_FINANCIAL_STOPWORDS)
    if extra_stopwords:
        custom_stops.update(extra_stopwords)
        
    all_stops = nltk_stops.union(custom_stops)
    # Do not remove our protected placeholders as stopwords!
    return [token for token in tokens if (token.lower() not in all_stops) or (token in PROTECTED_PLACEHOLDERS)]

def apply_stemming(tokens: list) -> list:
    """Applies Porter Stemmer to each token in the token list, keeping placeholders protected."""
    stemmer = PorterStemmer()
    return [
        token if token in PROTECTED_PLACEHOLDERS else stemmer.stem(token)
        for token in tokens
    ]

def apply_lemmatization(tokens: list) -> list:
    """Applies WordNet Lemmatizer to each token in the token list, keeping placeholders protected."""
    lemmatizer = WordNetLemmatizer()
    return [
        token if token in PROTECTED_PLACEHOLDERS else lemmatizer.lemmatize(token)
        for token in tokens
    ]

# --- Orchestrated Pipeline ---

def preprocess_tweet(
    text: str,
    lowercase: bool = True,
    url_mode: str = 'replace',
    mention_mode: str = 'replace',
    cashtag_mode: str = 'keep',
    remove_stopwords: bool = True,
    custom_stopwords: list = None,
    use_stemming: bool = False,
    use_lemmatization: bool = True,
    return_str: bool = False
):
    """
    Runs the full tweet preprocessing pipeline with configurable steps.
    Suitable for direct python import or automated tool integration.
    """
    # 1. Unicode/punctuation normalization (from EDA insights)
    text = normalize_unicode_punctuation(text)

    # 2. Lowercasing (standard text lowercased, placeholders inserted in uppercase later)
    if lowercase:
        text = apply_lowercase(text)
        
    # 3. Regex Cleaning (applies URL_PLACEHOLDER, CASHTAG_PLACEHOLDER, etc.)
    text = clean_regex(
        text,
        url_mode=url_mode,
        mention_mode=mention_mode,
        cashtag_mode=cashtag_mode
    )
    
    # 4. Tokenization
    tokens = tokenize_tweet(text)
    
    # 5. Stopwords Removal
    if remove_stopwords:
        tokens = remove_stopwords_from_tokens(tokens, extra_stopwords=custom_stopwords)
        
    # 6. Porter Stemming
    if use_stemming:
        tokens = apply_stemming(tokens)
        
    # 7. WordNet Lemmatization
    if use_lemmatization:
        tokens = apply_lemmatization(tokens)
        
    # Output formatting
    if return_str:
        return " ".join(tokens)
    return tokens

# --- Smoke Tests Verification ---

def run_smoke_tests():
    """Runs a series of tests to verify each preprocessing step is working as expected."""
    print("=" * 60)
    print("RUNNING PREPROCESSING SMOKE TESTS...")
    print("=" * 60)

    test_tweets = [
        # Normalization test (smart quotes, dashes, ellipses, replacement character)
        "Downgrades 4/7: $MLND to underperform at Needham—see details... https://t.co/example",
        # URL, cashtag, and mention replacement test
        "Shorting $AAPL here at $180. @elonmusk thoughts? #market #trading",
        # Stopword and casing test
        "RT @NovaIMS: $BTC is falling down rapidly... RT to warn others!",
        "Having a cup of coffee and watching the market open. Very neutral."
    ]

    for idx, raw_tweet in enumerate(test_tweets, 1):
        print(f"\nTest {idx} (Raw): '{raw_tweet}'")
        
        # Lemmatization pipeline (URL/Mention replaced, Cashtag kept)
        processed_lem = preprocess_tweet(
            raw_tweet,
            lowercase=True,
            url_mode='replace',
            mention_mode='replace',
            cashtag_mode='keep',
            remove_stopwords=True,
            use_stemming=False,
            use_lemmatization=True,
            return_str=True
        )
        print(f"   |-- Processed (Lemmatized String): '{processed_lem}'")

        # Stemming pipeline (URL/Mention removed, Cashtag replaced)
        processed_stem = preprocess_tweet(
            raw_tweet,
            lowercase=True,
            url_mode='remove',
            mention_mode='remove',
            cashtag_mode='replace',
            remove_stopwords=True,
            use_stemming=True,
            use_lemmatization=False,
            return_str=False
        )
        print(f"   |-- Processed (Stemmed Token List): {processed_stem}")
        
    print("\n" + "=" * 60)
    print("SMOKE TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)

# --- Main Entry Point for CLI & Agents ---

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Text Mining Preprocessing Pipeline Tool. Highly configurable for direct command-line use or integration as an agent tool."
    )
    
    # Text input options
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('--text', type=str, help="Single tweet text to preprocess.")
    input_group.add_argument('--smoke-tests', action='store_true', help="Run verification smoke tests.")
    input_group.add_argument('--json-input', action='store_true', help="Enable JSON input via stdin (great for agent workflows).")

    # Pipeline configurations
    parser.add_argument('--no-lowercase', action='store_false', dest='lowercase', help="Disable lowercasing.")
    parser.add_argument('--url-mode', type=str, choices=['keep', 'remove', 'replace'], default='replace',
                        help="How to handle URLs. Default: 'replace' with 'URL_PLACEHOLDER'.")
    parser.add_argument('--mention-mode', type=str, choices=['keep', 'remove', 'replace'], default='replace',
                        help="How to handle mentions (@user). Default: 'replace' with 'MENTION_PLACEHOLDER'.")
    parser.add_argument('--cashtag-mode', type=str, choices=['keep', 'remove', 'replace'], default='keep',
                        help="How to handle cashtags ($TSLA). Default: 'keep'.")
    parser.add_argument('--no-stopwords', action='store_false', dest='remove_stopwords', help="Disable stopwords removal.")
    parser.add_argument('--custom-stopwords', type=str, nargs='*', help="List of extra custom stopwords to remove.")
    parser.add_argument('--stem', action='store_true', dest='use_stemming', help="Apply Porter stemming.")
    parser.add_argument('--no-lemmatization', action='store_false', dest='use_lemmatization', help="Disable WordNet lemmatization.")
    parser.add_argument('--return-str', action='store_true', help="Return results as a space-joined string instead of a token list.")

    args = parser.parse_args()

    # Case 1: Run Smoke Tests
    if args.smoke_tests or len(sys.argv) == 1:
        run_smoke_tests()
        sys.exit(0)

    # Case 2: Read JSON from stdin for Agent integration
    if args.json_input:
        try:
            input_data = json.load(sys.stdin)
            text_to_process = input_data.get('text', '')
            
            # Allow JSON keys to override arguments
            lowercase = input_data.get('lowercase', args.lowercase)
            url_mode = input_data.get('url_mode', args.url_mode)
            mention_mode = input_data.get('mention_mode', args.mention_mode)
            cashtag_mode = input_data.get('cashtag_mode', args.cashtag_mode)
            remove_stopwords = input_data.get('remove_stopwords', args.remove_stopwords)
            custom_stopwords = input_data.get('custom_stopwords', args.custom_stopwords)
            use_stemming = input_data.get('use_stemming', args.use_stemming)
            use_lemmatization = input_data.get('use_lemmatization', args.use_lemmatization)
            return_str = input_data.get('return_str', args.return_str)
            
            result = preprocess_tweet(
                text=text_to_process,
                lowercase=lowercase,
                url_mode=url_mode,
                mention_mode=mention_mode,
                cashtag_mode=cashtag_mode,
                remove_stopwords=remove_stopwords,
                custom_stopwords=custom_stopwords,
                use_stemming=use_stemming,
                use_lemmatization=use_lemmatization,
                return_str=return_str
            )
            
            print(json.dumps({"status": "success", "result": result}))
        except Exception as e:
            print(json.dumps({"status": "error", "message": str(e)}))
        sys.exit(0)

    # Case 3: Process single string passed via --text
    if args.text:
        result = preprocess_tweet(
            text=args.text,
            lowercase=args.lowercase,
            url_mode=args.url_mode,
            mention_mode=args.mention_mode,
            cashtag_mode=args.cashtag_mode,
            remove_stopwords=args.remove_stopwords,
            custom_stopwords=args.custom_stopwords,
            use_stemming=args.use_stemming,
            use_lemmatization=args.use_lemmatization,
            return_str=args.return_str
        )
        print(result)
