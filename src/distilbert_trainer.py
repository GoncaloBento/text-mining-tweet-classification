#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TM-022: HF Trainer skeleton — tokeniser, dataloaders, and Trainer config.
Validates the full pipeline on 500 samples and caches tokenised datasets to disk.
"""

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import numpy as np
import pandas as pd
from pathlib import Path
from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, f1_score

from src.train_val_split import stratified_split
from src.evaluate import log_model_run, compute_metrics
from src.config import SEED


MODEL_NAME = "distilbert-base-uncased"
DATA_PATH = "data/train.csv"
CACHE_DIR = Path("outputs/distilbert_cache")
NUM_LABELS = 3  # Bearish=0, Bullish=1, Neutral=2
LABEL2ID = {"Bearish": 0, "Bullish": 1, "Neutral": 2}
ID2LABEL = {0: "Bearish", 1: "Bullish", 2: "Neutral"}


def load_data(n_samples: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    if n_samples:
        df = df.sample(n=n_samples, random_state=SEED).reset_index(drop=True)
    return df


def build_hf_datasets(
    tokenizer: DistilBertTokenizerFast,
    X_train: pd.Series,
    X_val: pd.Series,
    y_train: pd.Series,
    y_val: pd.Series,
    cache_dir: Path,
) -> tuple[Dataset, Dataset]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_cache = cache_dir / "train"
    val_cache = cache_dir / "val"

    def _tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=128)

    def _build(texts, labels, cache_path):
        if cache_path.exists():
            print(f"[CACHE] Loading from {cache_path}")
            return Dataset.load_from_disk(str(cache_path))
        ds = Dataset.from_dict({"text": texts.tolist(), "label": labels.tolist()})
        ds = ds.map(_tokenize, batched=True, remove_columns=["text"])
        ds.save_to_disk(str(cache_path))
        print(f"[CACHE] Saved to {cache_path}")
        return ds

    train_ds = _build(X_train, y_train, train_cache)
    val_ds = _build(X_val, y_val, val_cache)
    return train_ds, val_ds


def make_compute_metrics(owner: str = "La Feria", notes: str = ""):
    def _compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        metrics = compute_metrics(labels, preds)
        log_model_run(
            model_name="DistilBERT",
            feature_desc="HF fine-tune",
            metrics=metrics,
            params=f"model={MODEL_NAME}, max_length=128",
            owner=owner,
            notes=notes,
        )
        return {
            "accuracy": metrics["accuracy"],
            "f1_macro": metrics["f1_macro"],
        }
    return _compute_metrics


def build_trainer(
    model: DistilBertForSequenceClassification,
    tokenizer: DistilBertTokenizerFast,
    train_ds: Dataset,
    val_ds: Dataset,
    output_dir: str = "outputs/distilbert_checkpoints",
    learning_rate: float = 2e-5,
    per_device_batch_size: int = 16,
    num_epochs: int = 3,
    notes: str = "",
) -> Trainer:
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        seed=SEED,
        logging_steps=50,
        report_to="none",
    )

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=make_compute_metrics(notes=notes),
    )


def run_trainer(n_samples: int | None = 500, notes: str = "TM-022 skeleton validation") -> None:
    print(f"[DATA] Loading {'all' if n_samples is None else n_samples} samples ...")
    df = load_data(n_samples=n_samples)
    X_train, X_val, y_train, y_val = stratified_split(df)
    print(f"[SPLIT] train={len(X_train)}, val={len(X_val)}")

    print(f"[TOKENIZER] Loading {MODEL_NAME} ...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    train_ds, val_ds = build_hf_datasets(
        tokenizer, X_train, X_val, y_train, y_val, CACHE_DIR
    )
    print(f"[DATASET] train={len(train_ds)} rows, val={len(val_ds)} rows")

    print("[MODEL] Loading DistilBertForSequenceClassification ...")
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        train_ds=train_ds,
        val_ds=val_ds,
        notes=notes,
    )

    print("[TRAIN] Starting training ...")
    trainer.train()
    print("[EVAL] Final evaluation ...")
    trainer.evaluate()
    print("[OK] TM-022 complete.")


if __name__ == "__main__":
    run_trainer()
