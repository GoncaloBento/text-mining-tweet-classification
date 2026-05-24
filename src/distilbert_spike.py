#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TM-017: Hugging Face + DistilBERT environment spike.
Validates tokenizer loading and GPU access; tokenizes 200 samples as a sanity check.
No model training — environment validation only.
"""

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import pandas as pd
import torch
from transformers import DistilBertTokenizerFast


MODEL_NAME = "distilbert-base-uncased"
DATA_PATH = "data/train.csv"
N_SAMPLES = 200


def check_gpu() -> str:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"[GPU] CUDA available — {name}")
        return "cuda"
    print("[GPU] No CUDA device found — using CPU")
    return "cpu"


def load_tokenizer() -> DistilBertTokenizerFast:
    print(f"[TOKENIZER] Loading {MODEL_NAME} ...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    print(f"[TOKENIZER] Vocab size: {tokenizer.vocab_size}")
    return tokenizer


def tokenize_samples(tokenizer: DistilBertTokenizerFast, texts: list[str]) -> dict:
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    return encoded


def run_spike() -> None:
    device = check_gpu()

    tokenizer = load_tokenizer()

    df = pd.read_csv(DATA_PATH)
    sample_texts = df["text"].head(N_SAMPLES).tolist()
    print(f"[DATA] Loaded {len(sample_texts)} samples from {DATA_PATH}")

    encoded = tokenize_samples(tokenizer, sample_texts)
    input_ids = encoded["input_ids"].to(device)

    print(f"[SANITY] input_ids shape : {tuple(input_ids.shape)}")
    print(f"[SANITY] attention_mask shape: {tuple(encoded['attention_mask'].shape)}")
    print(f"[SANITY] Sample tokens (first row): {input_ids[0, :10].tolist()} ...")
    print("[OK] TM-017 spike complete — environment is ready for DistilBERT fine-tuning.")


if __name__ == "__main__":
    run_spike()
