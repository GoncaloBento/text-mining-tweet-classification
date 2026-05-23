#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
tests/test_experiment.py

Unit tests for the modular and idempotent experiment pipeline in src/experiment.py and src/evaluate.py.
Verifies pipeline execution, metric computation, and logging idempotence.
"""

import os
import csv
import unittest
import pandas as pd
from sklearn.datasets import make_classification

from src.experiment import run_tfidf_pipeline, run_model_pipeline
from src.evaluate import log_model_run, RESULTS_CSV_PATH, compute_metrics

class TestExperimentPipeline(unittest.TestCase):

    def setUp(self):
        """Set up dummy data and ensure clean results.csv testing state."""
        # Backup existing results.csv if it exists
        self.backup_path = RESULTS_CSV_PATH + ".bak"
        if os.path.exists(RESULTS_CSV_PATH):
            if os.path.exists(self.backup_path):
                os.remove(self.backup_path)
            os.rename(RESULTS_CSV_PATH, self.backup_path)

        # Generate standard mock data
        self.X_train = pd.Series([
            "extremely bullish on stock market today",
            "bearish outlook on tech stocks price target cuts",
            "neutral statement regarding shares value",
            "bullish momentum in trade deal",
            "underperform alert from analysts",
            "neutral results and earnings release call"
        ] * 5)
        
        self.X_val = pd.Series([
            "extremely bullish momentum on stocks price",
            "bearish target price downgrade",
            "neutral corporate announcement"
        ] * 2)
        
        self.y_train = pd.Series([1, 0, 2, 1, 0, 2] * 5)
        self.y_val = pd.Series([1, 0, 2] * 2)

    def tearDown(self):
        """Clean up test logs and restore backup results.csv."""
        if os.path.exists(RESULTS_CSV_PATH):
            os.remove(RESULTS_CSV_PATH)
            
        # Restore backup
        if os.path.exists(self.backup_path):
            os.rename(self.backup_path, RESULTS_CSV_PATH)

    def test_run_tfidf_pipeline(self):
        """Test that the core TF-IDF pipeline trains and returns metrics successfully."""
        vocab_size, metrics = run_tfidf_pipeline(
            self.X_train, self.X_val, self.y_train, self.y_val,
            ngram_range=(1, 1), min_df=1, max_features=None,
            feature_desc="Test Pipeline"
        )
        
        self.assertGreater(vocab_size, 0)
        self.assertIn("accuracy", metrics)
        self.assertIn("f1_macro", metrics)
        self.assertIsInstance(metrics["accuracy"], float)

    def test_run_model_pipeline(self):
        """Test that the general run_model_pipeline helper trains and evaluates a model successfully."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.neighbors import KNeighborsClassifier
        
        # Vectorize
        vec = TfidfVectorizer()
        X_train_vec = vec.fit_transform(self.X_train)
        X_val_vec = vec.transform(self.X_val)
        
        # Run general model pipeline
        metrics = run_model_pipeline(
            X_train_vec, X_val_vec, self.y_train, self.y_val,
            model=KNeighborsClassifier(n_neighbors=3),
            model_name="Test KNN",
            feature_desc="Test Space",
            params_str="n_neighbors=3"
        )
        
        self.assertIn("accuracy", metrics)
        self.assertIn("f1_macro", metrics)
        self.assertIsInstance(metrics["accuracy"], float)

    def test_logging_idempotence(self):
        """Test that log_model_run is fully idempotent and does not create duplicate rows."""
        model_name = "Test Idempotent Model"
        feature_desc = "Test TF-IDF (1,2)"
        params = "ngram_range=(1,2), min_df=2"
        dummy_metrics = {"accuracy": 0.80, "precision_macro": 0.75, "recall_macro": 0.75, "f1_macro": 0.75}

        # First run: should create new row
        log_model_run(model_name, feature_desc, dummy_metrics, params)
        self.assertTrue(os.path.exists(RESULTS_CSV_PATH))
        
        # Read file row count (excluding header)
        with open(RESULTS_CSV_PATH, 'r', encoding='utf-8') as f:
            rows_after_first = list(csv.reader(f))
        
        self.assertEqual(len(rows_after_first), 2) # Header + 1 row

        # Second run with identical parameters: should update existing row (no duplicates!)
        updated_metrics = {"accuracy": 0.85, "precision_macro": 0.80, "recall_macro": 0.80, "f1_macro": 0.80}
        log_model_run(model_name, feature_desc, updated_metrics, params)

        with open(RESULTS_CSV_PATH, 'r', encoding='utf-8') as f:
            rows_after_second = list(csv.reader(f))
            
        # Row count should STILL be 2 (Header + 1 row)
        self.assertEqual(len(rows_after_second), 2)
        
        # Read the metrics from the row to ensure they were updated to 0.8500
        headers = rows_after_second[0]
        data_row = dict(zip(headers, rows_after_second[1]))
        self.assertEqual(data_row["accuracy"], "0.8500")

        # Third run with DIFFERENT parameters: should append a new row
        different_params = "ngram_range=(1,2), min_df=5"
        log_model_run(model_name, feature_desc, dummy_metrics, different_params)

        with open(RESULTS_CSV_PATH, 'r', encoding='utf-8') as f:
            rows_after_third = list(csv.reader(f))
            
        # Row count should now be 3 (Header + 2 rows)
        self.assertEqual(len(rows_after_third), 3)

if __name__ == '__main__':
    unittest.main()
