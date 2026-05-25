import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem import PorterStemmer, WordNetLemmatizer
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

from src.config import (
    SEED, VAL_SIZE, K_FOLD_N_SPLITS,
    URL_PATTERN, MENTION_PATTERN, CASHTAG_PATTERN,
    URL_PLACEHOLDER, MENTION_PLACEHOLDER, CASHTAG_PLACEHOLDER, PROTECTED_PLACEHOLDERS,
    FINANCIAL_STOPWORDS,
)
from src.utils import log_info, log_success, print_header


def _download_nltk_resources() -> None:
    resources = {
        'stopwords': 'corpora/stopwords',
        'punkt': 'tokenizers/punkt',
        'wordnet': 'corpora/wordnet',
        'omw-1.4': 'corpora/omw-1.4',
    }
    for name, path in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


_download_nltk_resources()

_SPACE_RE = re.compile(r'\s+')
_UNICODE_MAP = [
    (re.compile(r'[""„‟]'), '"'),
    (re.compile(r"['''']"), "'"),
    (re.compile(r'[‒–—―−]'), '-'),
    (re.compile(r'…'), '...'),
    (re.compile(r'[�]'), ' '),
]
_TOKENIZER = TweetTokenizer(preserve_case=True, reduce_len=True, strip_handles=False)

# ── Text preprocessing ────────────────────────────────────────────────────────

def normalize_unicode_punctuation(text: str) -> str:
    if not isinstance(text, str):
        return ""
    for pattern, replacement in _UNICODE_MAP:
        text = pattern.sub(replacement, text)
    return text


def clean_regex(
    text: str,
    url_mode: str = 'replace',
    mention_mode: str = 'replace',
    cashtag_mode: str = 'keep',
) -> str:
    if url_mode == 'remove':
        text = URL_PATTERN.sub('', text)
    elif url_mode == 'replace':
        text = URL_PATTERN.sub(URL_PLACEHOLDER, text)

    if mention_mode == 'remove':
        text = MENTION_PATTERN.sub('', text)
    elif mention_mode == 'replace':
        text = MENTION_PATTERN.sub(MENTION_PLACEHOLDER, text)

    if cashtag_mode == 'remove':
        text = CASHTAG_PATTERN.sub('', text)
    elif cashtag_mode == 'replace':
        text = CASHTAG_PATTERN.sub(CASHTAG_PLACEHOLDER, text)

    return _SPACE_RE.sub(' ', text).strip()


def tokenize_tweet(text: str) -> list:
    return _TOKENIZER.tokenize(text)


def remove_stopwords_from_tokens(tokens: list, extra_stopwords: list = None) -> list:
    all_stops = set(stopwords.words('english')) | FINANCIAL_STOPWORDS
    if extra_stopwords:
        all_stops.update(extra_stopwords)
    return [t for t in tokens if (t.lower() not in all_stops) or (t in PROTECTED_PLACEHOLDERS)]


def apply_stemming(tokens: list) -> list:
    stemmer = PorterStemmer()
    return [t if t in PROTECTED_PLACEHOLDERS else stemmer.stem(t) for t in tokens]


def apply_lemmatization(tokens: list) -> list:
    lemmatizer = WordNetLemmatizer()
    return [t if t in PROTECTED_PLACEHOLDERS else lemmatizer.lemmatize(t) for t in tokens]


def preprocess_tweet(
    text: str,
    lowercase: bool = True,
    url_mode: str = 'replace',
    mention_mode: str = 'replace',
    cashtag_mode: str = 'keep',
    remove_stopwords: bool = True,
    custom_stopwords: list = None,
    use_stemming: bool = False,
    use_lemmatization: bool = True,
    return_str: bool = False,
):
    """Full tweet preprocessing pipeline. Returns token list or joined string."""
    text = normalize_unicode_punctuation(text)
    if lowercase:
        text = text.lower()
    text = clean_regex(text, url_mode=url_mode, mention_mode=mention_mode, cashtag_mode=cashtag_mode)
    tokens = tokenize_tweet(text)
    if remove_stopwords:
        tokens = remove_stopwords_from_tokens(tokens, extra_stopwords=custom_stopwords)
    if use_stemming:
        tokens = apply_stemming(tokens)
    if use_lemmatization:
        tokens = apply_lemmatization(tokens)
    return " ".join(tokens) if return_str else tokens


# ── Train / val splitting (absorbed from train_val_split.py) ──────────────────

def stratified_split(
    dataset: pd.DataFrame,
    test_size: float = VAL_SIZE,
    seed: int = SEED,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    X, y = dataset['text'], dataset['label']
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def create_stratified_kfold(n_splits: int = K_FOLD_N_SPLITS, seed: int = SEED) -> StratifiedKFold:
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


# ── Smoke tests ───────────────────────────────────────────────────────────────

def run_smoke_tests() -> None:
    """Verifies each preprocessing step with representative tweets."""
    print_header("PREPROCESSING SMOKE TESTS")

    test_tweets = [
        "Downgrades 4/7: $MLND to underperform at Needham—see details... https://t.co/example",
        "Shorting $AAPL here at $180. @elonmusk thoughts? #market #trading",
        "RT @NovaIMS: $BTC is falling down rapidly... RT to warn others!",
        "Having a cup of coffee and watching the market open. Very neutral.",
    ]

    for idx, raw in enumerate(test_tweets, 1):
        log_info(f"Test {idx}: '{raw}'")
        lem = preprocess_tweet(raw, return_str=True)
        log_info(f"  Lemmatized : '{lem}'")
        stem = preprocess_tweet(raw, url_mode='remove', mention_mode='remove',
                                cashtag_mode='replace', use_stemming=True,
                                use_lemmatization=False, return_str=False)
        log_info(f"  Stemmed    : {stem}")

    log_success("Smoke tests complete.")
