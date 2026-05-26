import numpy as np
import pandas as pd
from pathlib import Path
import torch
from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

from src.preprocessing import stratified_split
from src.evaluate import log_model_run, compute_metrics
from src.config import (
    SEED, TRAIN_CSV_PATH as DATA_PATH, TEST_CSV_PATH, NUM_LABELS, LABEL2ID, ID2LABEL,
    DISTILBERT_MODEL_NAME as MODEL_NAME,
    DISTILBERT_CACHE_DIR, DISTILBERT_CHECKPOINT_DIR,
    DISTILBERT_N_SAMPLES_SPIKE as N_SAMPLES_SPIKE,
)
from src.utils import log_info, log_success


# ── Environment helpers ──────────────────────────────────────────────────────

def check_gpu() -> str:
    if torch.cuda.is_available():
        log_info(f"CUDA available — {torch.cuda.get_device_name(0)}")
        return "cuda"
    log_info("No CUDA device found — using CPU")
    return "cpu"


def load_tokenizer() -> DistilBertTokenizerFast:
    log_info(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    log_info(f"Vocab size: {tokenizer.vocab_size}")
    return tokenizer


def tokenize_samples(tokenizer: DistilBertTokenizerFast, texts: list[str]) -> dict:
    return tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")


def run_spike() -> None:
    device = check_gpu()
    tokenizer = load_tokenizer()

    sample_texts = pd.read_csv(DATA_PATH)["text"].head(N_SAMPLES_SPIKE).tolist()
    log_info(f"Loaded {len(sample_texts)} samples from {DATA_PATH}")

    encoded = {k: v.to(device) for k, v in tokenize_samples(tokenizer, sample_texts).items()}

    log_info(f"input_ids shape        : {tuple(encoded['input_ids'].shape)}")
    log_info(f"attention_mask shape   : {tuple(encoded['attention_mask'].shape)}")
    log_info(f"Sample tokens (row 0)  : {encoded['input_ids'][0, :10].tolist()} ...")
    log_success("Spike complete — environment is ready for DistilBERT fine-tuning.")


# ── Training pipeline ─────────────────────────────────────────────────────────

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

    def _tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=128)

    def _build(texts, labels, cache_path):
        if cache_path.exists():
            log_info(f"Loading dataset from cache: {cache_path}")
            return Dataset.load_from_disk(str(cache_path))
        ds = Dataset.from_dict({"text": texts.tolist(), "label": labels.tolist()})
        ds = ds.map(_tokenize, batched=True, remove_columns=["text"])
        ds.save_to_disk(str(cache_path))
        log_info(f"Dataset cached to: {cache_path}")
        return ds

    train_ds = _build(X_train, y_train, cache_dir / "train")
    val_ds   = _build(X_val,   y_val,   cache_dir / "val")
    return train_ds, val_ds


def make_compute_metrics(owner: str = "", notes: str = ""):
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
        return {"accuracy": metrics["accuracy"], "f1_macro": metrics["f1_macro"]}
    return _compute_metrics


def build_trainer(
    model: DistilBertForSequenceClassification,
    tokenizer: DistilBertTokenizerFast,
    train_ds: Dataset,
    val_ds: Dataset,
    output_dir: str | None = None,
    learning_rate: float = 2e-5,
    per_device_batch_size: int = 16,
    num_epochs: int = 3,
    owner: str = "",
    notes: str = "",
) -> Trainer:
    training_args = TrainingArguments(
        output_dir=output_dir or str(Path(DISTILBERT_CHECKPOINT_DIR) / MODEL_NAME.replace("/", "_")),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        seed=SEED,
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
        compute_metrics=make_compute_metrics(owner=owner, notes=notes),
    )


def run_trainer(n_samples: int | None = 500, owner: str = "", notes: str = "") -> Trainer:
    log_info(f"Loading {'all' if n_samples is None else n_samples} samples ...")
    df = load_data(n_samples=n_samples)
    X_train, X_val, y_train, y_val = stratified_split(df)
    log_info(f"Split — train={len(X_train)}, val={len(X_val)}")

    tokenizer = load_tokenizer()

    # Use a sample-size-specific cache dir so a 500-sample spike doesn't get
    # served when we ask for the full dataset.
    cache_suffix = "full" if n_samples is None else f"n{n_samples}"
    cache_dir = Path(DISTILBERT_CACHE_DIR) / cache_suffix
    train_ds, val_ds = build_hf_datasets(tokenizer, X_train, X_val, y_train, y_val, cache_dir)
    log_info(f"Datasets — train={len(train_ds)} rows, val={len(val_ds)} rows")

    log_info("Loading DistilBertForSequenceClassification ...")
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS, id2label=ID2LABEL, label2id=LABEL2ID,
    )

    trainer = build_trainer(model=model, tokenizer=tokenizer,
                            train_ds=train_ds, val_ds=val_ds,
                            owner=owner, notes=notes)

    log_info("Starting training ...")
    trainer.train()
    log_info("Running final evaluation ...")
    trainer.evaluate()
    log_success("Training complete.")
    return trainer


def predict_test_set(
    trainer: Trainer,
    tokenizer: DistilBertTokenizerFast,
    test_csv_path: str = TEST_CSV_PATH,
) -> np.ndarray:
    log_info(f"Loading test set from {test_csv_path}")
    test_df = pd.read_csv(test_csv_path)
    log_info(f"Test rows: {len(test_df)}")

    def _tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=128)

    test_ds = Dataset.from_dict({"text": test_df["text"].tolist()})
    test_ds = test_ds.map(_tokenize, batched=True, remove_columns=["text"])

    preds = trainer.predict(test_ds)
    labels = np.argmax(preds.predictions, axis=-1)
    log_success(f"Generated {len(labels)} test predictions.")
    return labels
