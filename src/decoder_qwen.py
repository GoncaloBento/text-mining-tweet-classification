from __future__ import annotations

import re

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import SEED, TRAIN_CSV_PATH, NUM_LABELS, LABEL_NAMES, QWEN_MODEL_NAME
from src.evaluate import compute_metrics, log_model_run
from src.preprocessing import stratified_split
from src.utils import log_info, log_success, log_warning


LABEL_TO_ID = {v.lower(): k for k, v in LABEL_NAMES.items()}
MAJORITY_FALLBACK = 2  # Neutral — dominant class (~62 % of train set)

SYSTEM_PROMPT = (
    "You are a financial sentiment classifier for tweets about stocks. "
    "Each tweet is one of three classes: Bearish, Bullish, or Neutral. "
    "Reply with exactly ONE word — the class name — and nothing else."
)


# ── Model loading ────────────────────────────────────────────────────────────

def load_qwen(
    device: str | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype is None:
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

    log_info(f"Loading tokenizer: {QWEN_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_NAME)
    # Decoder-only models need left padding so the last token == the prediction position.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log_info(f"Loading model on {device} with dtype={dtype} ...")
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_NAME,
        torch_dtype=dtype,
        device_map=device,
    )
    model.eval()
    log_success("Qwen ready.")
    return tokenizer, model


# ── Prompt construction ──────────────────────────────────────────────────────

def _sample_few_shot_examples(train_df: pd.DataFrame, n_per_class: int, seed: int) -> list[tuple[str, str]]:
    examples: list[tuple[str, str]] = []
    rng = np.random.RandomState(seed)
    for label_id, label_name in LABEL_NAMES.items():
        pool = train_df.loc[train_df["label"] == label_id, "text"].tolist()
        idxs = rng.choice(len(pool), size=min(n_per_class, len(pool)), replace=False)
        for i in idxs:
            examples.append((pool[i], label_name))
    rng.shuffle(examples)  # avoid grouping by class
    return examples


def build_few_shot_messages(
    train_df: pd.DataFrame,
    n_per_class: int = 2,
    seed: int = SEED,
) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for text, label_name in _sample_few_shot_examples(train_df, n_per_class, seed):
        messages.append({"role": "user", "content": f"Tweet: {text}"})
        messages.append({"role": "assistant", "content": label_name})
    return messages


# ── Inference ────────────────────────────────────────────────────────────────

_LABEL_REGEX = re.compile(r"(bearish|bullish|neutral)", re.IGNORECASE)


def _parse_label(generated_text: str) -> int | None:
    m = _LABEL_REGEX.search(generated_text[:32])
    if m is None:
        return None
    return LABEL_TO_ID[m.group(1).lower()]


def _build_chat_string(tokenizer, few_shot_messages: list[dict], tweet: str) -> str:
    messages = list(few_shot_messages) + [{"role": "user", "content": f"Tweet: {tweet}"}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def classify_batch(
    tokenizer,
    model,
    few_shot_messages: list[dict],
    tweets: list[str],
    batch_size: int = 8,
    max_new_tokens: int = 4,
) -> tuple[list[int], int]:
    preds: list[int] = []
    parse_fails = 0
    device = next(model.parameters()).device

    for start in range(0, len(tweets), batch_size):
        chunk = tweets[start:start + batch_size]
        prompts = [_build_chat_string(tokenizer, few_shot_messages, t) for t in chunk]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        new_tokens = outputs[:, inputs["input_ids"].shape[1]:]
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

        for text in decoded:
            pid = _parse_label(text)
            if pid is None:
                parse_fails += 1
                preds.append(MAJORITY_FALLBACK)
            else:
                preds.append(pid)

        if (start // batch_size) % 10 == 0:
            log_info(f"  classified {start + len(chunk)}/{len(tweets)} (fails so far: {parse_fails})")

    return preds, parse_fails


def classify_tweet(tokenizer, model, few_shot_messages: list[dict], tweet: str) -> int:
    preds, _ = classify_batch(tokenizer, model, few_shot_messages, [tweet], batch_size=1)
    return preds[0]


# ── End-to-end evaluation ────────────────────────────────────────────────────

def run_qwen_eval(
    owner: str = "La Feria",
    notes: str = "TM-031 EXTRA WORK +1.0pt",
    n_per_class: int = 2,
    batch_size: int = 8,
    device: str | None = None,
    val_subset: int | None = None,
    tokenizer=None,
    model=None,
) -> dict:
    df = pd.read_csv(TRAIN_CSV_PATH)
    X_train, X_val, y_train, y_val = stratified_split(df)
    train_df = pd.DataFrame({"text": X_train.values, "label": y_train.values})

    if val_subset is not None:
        X_val = X_val.iloc[:val_subset]
        y_val = y_val.iloc[:val_subset]

    if tokenizer is None or model is None:
        tokenizer, model = load_qwen(device=device)
    few_shot_messages = build_few_shot_messages(train_df, n_per_class=n_per_class, seed=SEED)
    n_shots = n_per_class * NUM_LABELS

    log_info(f"Classifying {len(X_val)} val tweets with {n_shots}-shot Qwen ...")
    preds, parse_fails = classify_batch(
        tokenizer, model, few_shot_messages, X_val.tolist(), batch_size=batch_size,
    )
    parse_fail_rate = parse_fails / max(len(preds), 1)
    if parse_fail_rate > 0:
        log_warning(f"Parse-failure rate: {parse_fail_rate:.2%} ({parse_fails}/{len(preds)})")

    metrics = compute_metrics(y_val.values, np.array(preds))

    log_model_run(
        model_name="Qwen2.5-1.5B-Instruct",
        feature_desc=f"few-shot {n_shots}-shot ({n_per_class}/class) chat template",
        metrics=metrics,
        params=f"dtype=bf16, greedy, max_new_tokens=4, seed={SEED}, batch={batch_size}",
        owner=owner,
        notes=f"{notes}; parse_fail_rate={parse_fail_rate:.2%}",
    )
    log_success(f"Qwen run logged. f1_macro={metrics['f1_macro']:.4f}")
    return metrics
