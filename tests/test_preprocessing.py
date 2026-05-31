#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
tests/test_preprocessing.py

Dedicated unit tests for the text preprocessing pipeline in src/preprocessing.py.
Verifies each modular function (normalization, regex cleaning, stopwords,
tokenization, stemming, lemmatization) as well as the end-to-end pipeline.
"""

import unittest
from src.preprocessing import (
    normalize_unicode_punctuation,
    clean_regex,
    tokenize_tweet,
    remove_stopwords_from_tokens,
    apply_stemming,
    apply_lemmatization,
    preprocess_tweet,
    URL_PLACEHOLDER,
    MENTION_PLACEHOLDER,
    CASHTAG_PLACEHOLDER
)

class TestTweetPreprocessing(unittest.TestCase):

    def test_normalize_unicode_punctuation(self):
        """Test normalization of smart quotes, smart apostrophes, em-dashes, and ellipses."""
        # Smart double quotes
        self.assertEqual(normalize_unicode_punctuation('“underperform”'), '"underperform"')
        # Smart apostrophes
        self.assertEqual(normalize_unicode_punctuation("Netflix’s"), "Netflix's")
        # Long em-dashes and en-dashes
        self.assertEqual(normalize_unicode_punctuation("Needham—see"), "Needham-see")
        # Ellipses
        self.assertEqual(normalize_unicode_punctuation("details…"), "details...")
        # Unicode replacement character (\uFFFD)
        self.assertEqual(normalize_unicode_punctuation("price tar\uFFFD https"), "price tar  https")

    def test_clean_regex_urls(self):
        """Test different URL modes: keep, remove, and replace."""
        raw_text = "Check out https://t.co/example for more details."
        # Keep URL
        self.assertEqual(
            clean_regex(raw_text, url_mode='keep'),
            "Check out https://t.co/example for more details."
        )
        # Remove URL
        self.assertEqual(
            clean_regex(raw_text, url_mode='remove'),
            "Check out for more details."
        )
        # Replace URL
        self.assertEqual(
            clean_regex(raw_text, url_mode='replace'),
            f"Check out {URL_PLACEHOLDER} for more details."
        )

    def test_clean_regex_mentions(self):
        """Test different user mention (@username) modes: keep, remove, and replace."""
        raw_text = "What do you think @elonmusk?"
        # Keep Mention
        self.assertEqual(
            clean_regex(raw_text, mention_mode='keep'),
            "What do you think @elonmusk?"
        )
        # Remove Mention
        self.assertEqual(
            clean_regex(raw_text, mention_mode='remove'),
            "What do you think ?"
        )
        # Replace Mention
        self.assertEqual(
            clean_regex(raw_text, mention_mode='replace'),
            f"What do you think {MENTION_PLACEHOLDER}?"
        )

    def test_clean_regex_cashtags(self):
        """Test different cashtag ($AAPL) modes: keep, remove, and replace."""
        raw_text = "Bullish on $TSLA and $AAPL today!"
        # Keep Cashtags
        self.assertEqual(
            clean_regex(raw_text, cashtag_mode='keep'),
            "Bullish on $TSLA and $AAPL today!"
        )
        # Remove Cashtags
        self.assertEqual(
            clean_regex(raw_text, cashtag_mode='remove'),
            "Bullish on and today!"
        )
        # Replace Cashtags
        self.assertEqual(
            clean_regex(raw_text, cashtag_mode='replace'),
            f"Bullish on {CASHTAG_PLACEHOLDER} and {CASHTAG_PLACEHOLDER} today!"
        )

    def test_tokenize_tweet(self):
        """Test tokenization using TweetTokenizer."""
        text = "Bullish on $TSLA! #market @NovaIMS"
        tokens = tokenize_tweet(text)
        expected = ["Bullish", "on", "$", "TSLA", "!", "#market", "@NovaIMS"]
        self.assertEqual(tokens, expected)

    def test_remove_stopwords_from_tokens(self):
        """Test removing standard and custom financial stopwords from a list of tokens."""
        tokens = ["rt", "the", "stock", "is", "bullish", "on", "$TSLA"]
        # 'rt' and 'stock' are custom financial stopwords, 'the', 'is', 'on' are NLTK stopwords
        cleaned = remove_stopwords_from_tokens(tokens)
        self.assertEqual(cleaned, ["bullish", "$TSLA"])

        # Test custom stopwords override
        extra = ["bullish"]
        cleaned_extra = remove_stopwords_from_tokens(tokens, extra_stopwords=extra)
        self.assertEqual(cleaned_extra, ["$TSLA"])

    def test_apply_stemming_protected(self):
        """Test that Porter Stemming stems standard words but leaves protected placeholders intact."""
        tokens = ["extremely", "bullish", URL_PLACEHOLDER, MENTION_PLACEHOLDER, CASHTAG_PLACEHOLDER]
        stemmed = apply_stemming(tokens)
        expected = ["extrem", "bullish", URL_PLACEHOLDER, MENTION_PLACEHOLDER, CASHTAG_PLACEHOLDER]
        self.assertEqual(stemmed, expected)

    def test_apply_lemmatization_protected(self):
        """Test that WordNet Lemmatizer lemmatizes standard words but leaves placeholders intact."""
        tokens = ["thoughts", "downgrades", URL_PLACEHOLDER, MENTION_PLACEHOLDER, CASHTAG_PLACEHOLDER]
        lemmatized = apply_lemmatization(tokens)
        expected = ["thought", "downgrade", URL_PLACEHOLDER, MENTION_PLACEHOLDER, CASHTAG_PLACEHOLDER]
        self.assertEqual(lemmatized, expected)

    def test_preprocess_tweet_pipeline(self):
        """Test the full orchestrated preprocess_tweet pipeline under typical configuration."""
        raw_tweet = "RT @NovaIMS: $BTC is falling down rapidly... Check out https://t.co/example!"
        
        # Lemmatized list output
        processed_tokens = preprocess_tweet(
            raw_tweet,
            lowercase=True,
            url_mode='replace',
            mention_mode='replace',
            cashtag_mode='keep',
            remove_stopwords=True,
            use_stemming=False,
            use_lemmatization=True,
            return_str=False
        )
        expected_tokens = [
            MENTION_PLACEHOLDER, ":", "$", "btc", "falling", "rapidly", "...", "warn", "others", "!", URL_PLACEHOLDER, "!"
        ]
        # Check that protected placeholders are in the output list intact
        self.assertIn(URL_PLACEHOLDER, processed_tokens)
        self.assertIn(MENTION_PLACEHOLDER, processed_tokens)
        
        # Return as rejoined string
        processed_str = preprocess_tweet(
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
        self.assertIsInstance(processed_str, str)
        self.assertTrue(URL_PLACEHOLDER in processed_str)
        self.assertTrue(MENTION_PLACEHOLDER in processed_str)

if __name__ == '__main__':
    unittest.main()
