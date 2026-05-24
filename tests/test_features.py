#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
tests/test_features.py

Unit test suite to verify CountVectorizer and TfidfVectorizer pipeline correctness,
vocabulary consistency, and out-of-sample transformation dimensions.
"""

import os
import unittest
import scipy.sparse as sp

class TestFeatureVectorization(unittest.TestCase):
    
    def setUp(self):
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.outputs_dir = os.path.join(self.project_root, "outputs")
        
        self.train_bow_path = os.path.join(self.outputs_dir, "X_train_bow.npz")
        self.val_bow_path = os.path.join(self.outputs_dir, "X_val_bow.npz")
        self.train_tfidf_path = os.path.join(self.outputs_dir, "X_train_tfidf_uni.npz")
        self.val_tfidf_path = os.path.join(self.outputs_dir, "X_val_tfidf_uni.npz")
        self.train_tfidf_opt_path = os.path.join(self.outputs_dir, "X_train_tfidf_opt.npz")
        self.val_tfidf_opt_path = os.path.join(self.outputs_dir, "X_val_tfidf_opt.npz")

    def test_sparse_files_exist(self):
        """Asserts that all 6 sparse matrix files exist in outputs/ directory."""
        self.assertTrue(os.path.exists(self.train_bow_path), "X_train_bow.npz does not exist.")
        self.assertTrue(os.path.exists(self.val_bow_path), "X_val_bow.npz does not exist.")
        self.assertTrue(os.path.exists(self.train_tfidf_path), "X_train_tfidf_uni.npz does not exist.")
        self.assertTrue(os.path.exists(self.val_tfidf_path), "X_val_tfidf_uni.npz does not exist.")
        self.assertTrue(os.path.exists(self.train_tfidf_opt_path), "X_train_tfidf_opt.npz does not exist.")
        self.assertTrue(os.path.exists(self.val_tfidf_opt_path), "X_val_tfidf_opt.npz does not exist.")

    def test_matrix_shapes_and_sparsity(self):
        """Asserts that sparse CSR matrices load properly and match expected shapes."""
        # 1. Load matrices
        X_train_bow = sp.load_npz(self.train_bow_path)
        X_val_bow = sp.load_npz(self.val_bow_path)
        X_train_tfidf = sp.load_npz(self.train_tfidf_path)
        X_val_tfidf = sp.load_npz(self.val_tfidf_path)
        X_train_tfidf_opt = sp.load_npz(self.train_tfidf_opt_path)
        X_val_tfidf_opt = sp.load_npz(self.val_tfidf_opt_path)
        
        # 2. Check shapes
        # Train fold size must be 7634, Val fold size must be 1909
        # Both vocabularies fitted on train must be 12575 features
        self.assertEqual(X_train_bow.shape, (7634, 12575), "Train BoW shape mismatch.")
        self.assertEqual(X_val_bow.shape, (1909, 12575), "Val BoW shape mismatch.")
        self.assertEqual(X_train_tfidf.shape, (7634, 12575), "Train TF-IDF shape mismatch.")
        self.assertEqual(X_val_tfidf.shape, (1909, 12575), "Val TF-IDF shape mismatch.")
        
        self.assertEqual(X_train_tfidf_opt.shape[0], 7634, "Train Optimized TF-IDF shape mismatch (rows).")
        self.assertEqual(X_val_tfidf_opt.shape[0], 1909, "Val Optimized TF-IDF shape mismatch (rows).")
        self.assertEqual(X_train_tfidf_opt.shape[1], X_val_tfidf_opt.shape[1], "Opt TF-IDF feature sizes must match.")
        
        # 3. Check types
        self.assertTrue(sp.isspmatrix_csr(X_train_bow), "X_train_bow must be a CSR sparse matrix.")
        self.assertTrue(sp.isspmatrix_csr(X_val_bow), "X_val_bow must be a CSR sparse matrix.")
        self.assertTrue(sp.isspmatrix_csr(X_train_tfidf), "X_train_tfidf must be a CSR sparse matrix.")
        self.assertTrue(sp.isspmatrix_csr(X_val_tfidf), "X_val_tfidf must be a CSR sparse matrix.")
        self.assertTrue(sp.isspmatrix_csr(X_train_tfidf_opt), "X_train_tfidf_opt must be a CSR sparse matrix.")
        self.assertTrue(sp.isspmatrix_csr(X_val_tfidf_opt), "X_val_tfidf_opt must be a CSR sparse matrix.")

if __name__ == "__main__":
    unittest.main()
