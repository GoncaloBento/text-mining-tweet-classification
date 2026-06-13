"""Shared HuggingFace fine-tuning pipeline for sequence classification.

All four encoders (DistilBERT, FinBERT, Twitter-RoBERTa, DeBERTa-v3) share an
identical training loop and differ only in a handful of constants, captured in a
`TrainerSpec` and registered in `SPECS`. Pick a backbone by key:
`run_trainer("finbert", n_samples=1000)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.config import (
    SEED, TRAIN_CSV_PATH, TEST_CSV_PATH, NUM_LABELS, LABEL2ID, ID2LABEL,
    DISTILBERT_MODEL_NAME, DISTILBERT_CACHE_DIR, DISTILBERT_CHECKPOINT_DIR,
    FINBERT_MODEL_NAME, FINBERT_CACHE_DIR, FINBERT_CHECKPOINT_DIR,
    ROBERTA_MODEL_NAME, ROBERTA_CACHE_DIR, ROBERTA_CHECKPOINT_DIR,
    DEBERTA_MODEL_NAME, DEBERTA_CACHE_DIR, DEBERTA_CHECKPOINT_DIR,
)
from src.evaluate import log_model_run, compute_metrics
from src.preprocessing import stratified_split
from src.utils import log_info, log_success

MAX_LENGTH = 128


@dataclass(frozen=True)
class TrainerSpec:
    """Everything that distinguishes one encoder trainer from another."""

    display_name: str   # label used in the results leaderboard
    model_name: str     # HF hub id
    cache_dir: str      # tokenized-dataset cache root
    checkpoint_dir: str  # Trainer output root


# All four encoders differ only in config, so they live here as a registry.
# Add a new backbone by adding one entry.
SPECS: dict[str, TrainerSpec] = {
    "distilbert": TrainerSpec("DistilBERT", DISTILBERT_MODEL_NAME,
                              DISTILBERT_CACHE_DIR, DISTILBERT_CHECKPOINT_DIR),
    "finbert":    TrainerSpec("FinBERT", FINBERT_MODEL_NAME,
                              FINBERT_CACHE_DIR, FINBERT_CHECKPOINT_DIR),
    "roberta":    TrainerSpec("Twitter-RoBERTa", ROBERTA_MODEL_NAME,
                              ROBERTA_CACHE_DIR, ROBERTA_CHECKPOINT_DIR),
    "deberta":    TrainerSpec("deberta-v3-base", DEBERTA_MODEL_NAME,
                              DEBERTA_CACHE_DIR, DEBERTA_CHECKPOINT_DIR),
}


def load_tokenizer(spec: TrainerSpec) -> AutoTokenizer:
    log_info(f"Loading tokenizer: {spec.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(spec.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _tokenize_dataset(tokenizer, texts, labels=None, max_length: int = MAX_LENGTH) -> Dataset:
    """Build a tokenized HF Dataset from texts (and optional labels)."""
    data = {"text": list(texts)}
    if labels is not None:
        data["label"] = list(labels)
    return Dataset.from_dict(data).map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=max_length),
        batched=True,
        remove_columns=["text"],
    )


def build_hf_datasets(tokenizer, X_train, X_val, y_train, y_val, cache_dir: Path,
                      max_length: int = MAX_LENGTH):
    cache_dir.mkdir(parents=True, exist_ok=True)

    def _build(texts, labels, cache_path):
        if cache_path.exists():
            log_info(f"Loading dataset from cache: {cache_path}")
            return Dataset.load_from_disk(str(cache_path))
        ds = _tokenize_dataset(tokenizer, texts, labels, max_length=max_length)
        ds.save_to_disk(str(cache_path))
        log_info(f"Dataset cached to: {cache_path}")
        return ds

    return (_build(X_train, y_train, cache_dir / "train"),
            _build(X_val,   y_val,   cache_dir / "val"))


def make_compute_metrics(spec: TrainerSpec, notes: str = "", params: str = ""):
    def _compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        metrics = compute_metrics(labels, preds)
        log_model_run(
            model_name=spec.display_name,
            feature_desc="HF fine-tune",
            metrics=metrics,
            params=params or f"model={spec.model_name}, max_length={MAX_LENGTH}",
            notes=notes,
        )
        return {"accuracy": metrics["accuracy"], "f1_macro": metrics["f1_macro"]}
    return _compute_metrics


def build_trainer(spec: TrainerSpec, model, tokenizer, train_ds, val_ds, notes: str = "",
                  learning_rate: float = 2e-5, batch_size: int = 16,
                  seed: int = SEED, params: str = "") -> Trainer:
    training_args = TrainingArguments(
        output_dir=str(Path(spec.checkpoint_dir) / spec.model_name.replace("/", "_")),
        num_train_epochs=3,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        seed=seed,
        logging_steps=50,
        report_to="none",
    )
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=make_compute_metrics(spec, notes=notes, params=params),
    )


def run_trainer(model: str, n_samples: int | None = 500, notes: str = "",
                learning_rate: float = 2e-5, max_length: int = MAX_LENGTH,
                seed: int = SEED, batch_size: int = 16) -> Trainer:
    """Fine-tune one of the registered encoders.

    `model` is a key of SPECS: "distilbert", "finbert", "roberta", "deberta".
    `learning_rate`, `max_length`, `seed` and `batch_size` allow controlled
    variations; each combination gets its own leaderboard row and dataset cache.
    """
    spec = SPECS[model]

    log_info(f"Loading {'all' if n_samples is None else n_samples} samples ...")
    df = pd.read_csv(TRAIN_CSV_PATH)
    if n_samples:
        df = df.sample(n=n_samples, random_state=SEED).reset_index(drop=True)
    X_train, X_val, y_train, y_val = stratified_split(df)
    log_info(f"Split — train={len(X_train)}, val={len(X_val)}")

    tokenizer = load_tokenizer(spec)

    # Sample-size-specific cache dir so a 500-sample spike isn't served when we
    # ask for the full dataset. max_length variants get their own cache too.
    suffix = "full" if n_samples is None else f"n{n_samples}"
    if max_length != MAX_LENGTH:
        suffix += f"_len{max_length}"
    cache_dir = Path(spec.cache_dir) / suffix
    train_ds, val_ds = build_hf_datasets(tokenizer, X_train, X_val, y_train, y_val, cache_dir,
                                         max_length=max_length)
    log_info(f"Datasets — train={len(train_ds)} rows, val={len(val_ds)} rows")

    log_info(f"Loading {spec.model_name} sequence classifier ...")
    # ignore_mismatched_sizes=True safely re-initializes the classification head
    # for our project's 3-label schema.
    hf_model = AutoModelForSequenceClassification.from_pretrained(
        spec.model_name,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    params = (f"model={spec.model_name}, max_length={max_length}, lr={learning_rate}, "
              f"seed={seed}, n={'full' if n_samples is None else n_samples}")
    trainer = build_trainer(spec, hf_model, tokenizer, train_ds, val_ds, notes=notes,
                            learning_rate=learning_rate, batch_size=batch_size,
                            seed=seed, params=params)

    log_info("Starting training ...")
    trainer.train()
    log_info("Running final evaluation ...")
    trainer.evaluate()
    log_success("Training complete.")
    return trainer


def predict_test_set(trainer, tokenizer, test_csv_path: str = TEST_CSV_PATH) -> np.ndarray:
    log_info(f"Loading test set from {test_csv_path}")
    test_df = pd.read_csv(test_csv_path)
    log_info(f"Test rows: {len(test_df)}")

    test_ds = _tokenize_dataset(tokenizer, test_df["text"])
    preds = trainer.predict(test_ds)
    labels = np.argmax(preds.predictions, axis=-1)
    log_success(f"Generated {len(labels)} test predictions.")
    return labels
