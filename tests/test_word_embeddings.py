#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
tests/test_word_embeddings.py

Unit tests for the word embeddings utility functions in src/word_embeddings.py.
"""

import unittest
import numpy as np

from src.word_embeddings import calculate_oov, vectorize_document, vectorize_corpus

class TestWordEmbeddings(unittest.TestCase):

    def setUp(self):
        # Create a toy vocabulary and toy vectors mapping
        self.toy_vocab = {"apple": 0, "banana": 1, "cherry": 2}
        self.toy_vectors = {
            "apple": np.array([1.0, 2.0, 3.0]),
            "banana": np.array([4.0, 5.0, 6.0]),
            "cherry": np.array([7.0, 8.0, 9.0])
        }

    def test_calculate_oov_rates(self):
        # Validation tokenized sentences
        # 'date' is OOV, 'durian' is OOV. 'apple' and 'banana' are in vocab.
        tokenized_sentences = [
            ["apple", "banana"],
            ["apple", "date"],
            ["durian"]
        ]
        
        # Unique tokens in corpus: {'apple', 'banana', 'date', 'durian'} -> 4 unique
        # OOV unique tokens: {'date', 'durian'} -> 2 unique OOV
        # Unique OOV Rate: 2 / 4 = 50.0%
        
        # Total tokens in corpus: apple (2), banana (1), date (1), durian (1) -> 5 tokens
        # OOV tokens: date (1), durian (1) -> 2 OOV tokens
        # Token OOV Rate: 2 / 5 = 40.0%
        
        uniq_oov_rate, tok_oov_rate, oov_words = calculate_oov(tokenized_sentences, self.toy_vocab)
        
        self.assertAlmostEqual(uniq_oov_rate, 0.50)
        self.assertAlmostEqual(tok_oov_rate, 0.40)
        self.assertIn("date", oov_words)
        self.assertIn("durian", oov_words)
        self.assertNotIn("apple", oov_words)

    def test_vectorize_document_mean_pooling(self):
        # Case 1: All tokens are in vocabulary
        tokens_all_in = ["apple", "banana"]
        expected_mean = np.array([2.5, 3.5, 4.5])  # Mean of [1, 2, 3] and [4, 5, 6]
        pooled_vector = vectorize_document(tokens_all_in, self.toy_vocab, self.toy_vectors, vector_size=3)
        np.testing.assert_array_almost_equal(pooled_vector, expected_mean)

        # Case 2: Some tokens are OOV
        tokens_partial_oov = ["apple", "date"]
        expected_mean_partial = np.array([1.0, 2.0, 3.0])  # Only 'apple' vector is kept
        pooled_vector_partial = vectorize_document(tokens_partial_oov, self.toy_vocab, self.toy_vectors, vector_size=3)
        np.testing.assert_array_almost_equal(pooled_vector_partial, expected_mean_partial)

        # Case 3: All tokens are OOV (fallback zero vector)
        tokens_all_oov = ["date", "durian"]
        expected_zero = np.zeros(3)
        pooled_vector_oov = vectorize_document(tokens_all_oov, self.toy_vocab, self.toy_vectors, vector_size=3)
        np.testing.assert_array_almost_equal(pooled_vector_oov, expected_zero)

    def test_vectorize_corpus(self):
        tokenized_sentences = [
            ["apple", "banana"],
            ["date", "durian"]
        ]
        
        # Expected corpus vectors shape: (2, 3)
        corpus_vectors = vectorize_corpus(tokenized_sentences, self.toy_vectors, vector_size=3)
        
        self.assertEqual(corpus_vectors.shape, (2, 3))
        np.testing.assert_array_almost_equal(corpus_vectors[0], np.array([2.5, 3.5, 4.5]))
        np.testing.assert_array_almost_equal(corpus_vectors[1], np.zeros(3))

if __name__ == "__main__":
    unittest.main()
