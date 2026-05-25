import numpy as np
from gensim.models import Word2Vec
import gensim.downloader as api

from src.config import SEED
from src.utils import log_info, log_success, log_error


def _get_vocab_and_vectors(embedding_model):
    if hasattr(embedding_model, 'key_to_index'):
        return embedding_model.key_to_index, embedding_model
    if hasattr(embedding_model, 'wv') and hasattr(embedding_model.wv, 'key_to_index'):
        return embedding_model.wv.key_to_index, embedding_model.wv
    return embedding_model, embedding_model


def calculate_oov(tokenized_sentences: list, embedding_model) -> tuple[float, float, set]:
    """Returns (unique_oov_rate, token_oov_rate, oov_word_set)."""
    vocab, _ = _get_vocab_and_vectors(embedding_model)
    all_tokens = [tok for sent in tokenized_sentences for tok in sent]
    unique_tokens = set(all_tokens)
    if not unique_tokens:
        return 0.0, 0.0, set()
    oov_unique = {tok for tok in unique_tokens if tok not in vocab}
    oov_tokens = [tok for tok in all_tokens if tok not in vocab]
    return len(oov_unique) / len(unique_tokens), len(oov_tokens) / len(all_tokens), oov_unique


def vectorize_corpus(tokenized_sentences: list, embedding_model, vector_size: int = 100) -> np.ndarray:
    """Mean-pools token embeddings for each sentence in the corpus."""
    vocab, vectors = _get_vocab_and_vectors(embedding_model)
    result = []
    for sent in tokenized_sentences:
        valid = [vectors[tok] for tok in sent if tok in vocab]
        result.append(np.mean(valid, axis=0) if valid else np.zeros(vector_size))
    return np.array(result)


def train_word2vec(tokenized_sentences: list, vector_size: int = 100, window: int = 5, min_count: int = 2) -> Word2Vec:
    """Trains a Word2Vec model on the given tokenized corpus."""
    model = Word2Vec(
        sentences=tokenized_sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        seed=SEED,
        workers=4,
    )
    log_success(f"Word2Vec trained — vocab size: {len(model.wv.key_to_index)}")
    return model


def load_glove_twitter(dim: int = 100):
    """Loads GloVe-Twitter embeddings via gensim downloader (downloads ~400 MB if not cached)."""
    log_info(f"Loading glove-twitter-{dim} (downloads if not cached) ...")
    try:
        model = api.load(f"glove-twitter-{dim}")
        log_success(f"GloVe-Twitter-{dim} loaded — vocab size: {len(model.key_to_index)}")
        return model
    except Exception as e:
        log_error(f"Failed to load GloVe-Twitter-{dim}: {e}")
        raise
