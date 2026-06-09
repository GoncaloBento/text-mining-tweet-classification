import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from src.config import DISTILBERT_MODEL_NAME

_tokenizer = None
_model = None


def _load():
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_MODEL_NAME)
        _model = AutoModel.from_pretrained(DISTILBERT_MODEL_NAME)
        _model.eval()
    return _tokenizer, _model


def encoder_features(texts: list[str], pool: str = "mean", batch_size: int = 64) -> np.ndarray:
    tokenizer, model = _load()
    device = next(model.parameters()).device
    all_vecs = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**enc)
        hidden = out.last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1)

        if pool == "cls":
            vecs = hidden[:, 0, :]
        else:
            vecs = (hidden * mask).sum(dim=1) / mask.sum(dim=1)

        all_vecs.append(vecs.cpu().numpy())

    return np.vstack(all_vecs)
