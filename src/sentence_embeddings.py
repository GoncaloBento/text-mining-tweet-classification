import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from src.config import (
    TRAIN_CSV_PATH, 
    DISTILBERT_MODEL_NAME, DISTILBERT_CHECKPOINT_DIR, 
    FINBERT_MODEL_NAME, FINBERT_CHECKPOINT_DIR
)
from src.preprocessing import stratified_split
from src.utils import log_info, log_success

def get_best_checkpoint(model_name, checkpoint_dir):
    """Returns the path to the fine-tuned checkpoint if it exists, otherwise falls back to base model."""
    if os.path.exists(checkpoint_dir):
        checkpoints = [os.path.join(checkpoint_dir, d) for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint")]
        if checkpoints:
            best_ckpt = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1]
            log_info(f"Found fine-tuned checkpoint: {best_ckpt}")
            return best_ckpt
    log_info(f"Fine-tuned checkpoint not found locally. Using base model: {model_name}")
    return model_name

def extract_embeddings(texts, model_path, pooling="cls", batch_size=32):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_info(f"Loading {model_path} on {device}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    model.to(device)
    model.eval()

    all_embs = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc=f"Extracting {pooling} from {model_path}"):
            batch = texts[i:i+batch_size]
            encoded = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
            encoded = {k: v.to(device) for k, v in encoded.items()}
            
            out = model(**encoded)
            hidden = out.last_hidden_state  # (B, L, H)
            
            if pooling == "cls":
                emb = hidden[:, 0, :]
            elif pooling == "mean":
                mask = encoded["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
                sum_emb = torch.sum(hidden * mask, 1)
                sum_mask = torch.clamp(mask.sum(1), min=1e-9)
                emb = sum_emb / sum_mask
            else:
                raise ValueError("Pooling must be cls or mean")
                
            all_embs.append(emb.cpu().numpy())
            
    return np.vstack(all_embs)

def encoder_features(texts, pooling="cls", batch_size=32, model_path=None):
    """Sentence embeddings from a frozen (not fine-tuned) encoder, as a feature matrix."""
    if model_path is None:
        model_path = DISTILBERT_MODEL_NAME
    return extract_embeddings(list(texts), model_path, pooling=pooling, batch_size=batch_size)

def generate_and_save_embeddings():
    log_info("Loading train data...")
    df = pd.read_csv(TRAIN_CSV_PATH)
    X_train_raw, X_val_raw, y_train, y_val = stratified_split(df)
    
    train_texts = X_train_raw.tolist()
    val_texts = X_val_raw.tolist()
    
    out_dir = "outputs/embeddings"
    os.makedirs(out_dir, exist_ok=True)
    
    models_to_test = {
        "distilbert": get_best_checkpoint(DISTILBERT_MODEL_NAME, DISTILBERT_CHECKPOINT_DIR),
        "finbert": get_best_checkpoint(FINBERT_MODEL_NAME, FINBERT_CHECKPOINT_DIR)
    }
    
    for name, path in models_to_test.items():
        for pooling in ["cls", "mean"]:
            log_info(f"Processing {name} with {pooling} pooling...")
            
            train_path = os.path.join(out_dir, f"X_train_{name}_{pooling}.npy")
            val_path = os.path.join(out_dir, f"X_val_{name}_{pooling}.npy")
            
            if os.path.exists(train_path) and os.path.exists(val_path):
                log_info(f"Embeddings already exist for {name} ({pooling}). Skipping.")
                continue
                
            X_tr = extract_embeddings(train_texts, path, pooling=pooling)
            np.save(train_path, X_tr)
            
            X_va = extract_embeddings(val_texts, path, pooling=pooling)
            np.save(val_path, X_va)
            
    # Save labels
    np.save(os.path.join(out_dir, "y_train.npy"), y_train.values)
    np.save(os.path.join(out_dir, "y_val.npy"), y_val.values)
    log_success(f"All embeddings saved to {out_dir}")

if __name__ == "__main__":
    generate_and_save_embeddings()

