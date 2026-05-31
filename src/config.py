import re

# Reproducibility
SEED = 42

# Dataset splits
VAL_SIZE = 0.20
K_FOLD_N_SPLITS = 5

# Paths
TRAIN_CSV_PATH = "data/train.csv"
TEST_CSV_PATH = "data/test.csv"
RESULTS_CSV_PATH = "outputs/results.csv"
OUTPUT_PRED_PATH = "outputs/pred_best.csv"

# Error analysis output paths
CONF_MATRIX_PLOT_PATH = "outputs/confusion_matrix.png"
MISCLASSIFIED_TXT_PATH = "outputs/misclassified_report.txt"
MISCLASSIFIED_JSON_PATH = "outputs/misclassified_analysis.json"

# Labels
NUM_LABELS = 3
LABEL_NAMES = {0: "Bearish", 1: "Bullish", 2: "Neutral"}
LABEL2ID = {"Bearish": 0, "Bullish": 1, "Neutral": 2}
ID2LABEL = {0: "Bearish", 1: "Bullish", 2: "Neutral"}

# DistilBERT
DISTILBERT_MODEL_NAME = "distilbert-base-uncased"
DISTILBERT_N_SAMPLES_SPIKE = 200
DISTILBERT_CACHE_DIR = "outputs/distilbert_cache"
DISTILBERT_CHECKPOINT_DIR = "outputs/distilbert_checkpoints"

# Qwen decoder
QWEN_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# Twitter-RoBERTa
ROBERTA_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
ROBERTA_N_SAMPLES_SPIKE = 200
ROBERTA_CACHE_DIR = "outputs/roberta_cache"
ROBERTA_CHECKPOINT_DIR = "outputs/roberta_checkpoints"

# Feature matrix cache paths
BOW_TRAIN_PATH       = "outputs/X_train_bow.npz"
BOW_VAL_PATH         = "outputs/X_val_bow.npz"
TFIDF_UNI_TRAIN_PATH = "outputs/X_train_tfidf_uni.npz"
TFIDF_UNI_VAL_PATH   = "outputs/X_val_tfidf_uni.npz"
TFIDF_OPT_TRAIN_PATH = "outputs/X_train_tfidf_opt.npz"
TFIDF_OPT_VAL_PATH   = "outputs/X_val_tfidf_opt.npz"

# Leaderboard CSV schema
RESULTS_HEADERS = [
    "timestamp", "owner", "model_name", "feature_description",
    "accuracy", "precision_macro", "recall_macro", "f1_macro",
    "parameters", "f1_per_class", "notes",
]

# Preprocessing — placeholders
URL_PLACEHOLDER = "URL_PLACEHOLDER"
MENTION_PLACEHOLDER = "MENTION_PLACEHOLDER"
CASHTAG_PLACEHOLDER = "CASHTAG_PLACEHOLDER"
PROTECTED_PLACEHOLDERS = {URL_PLACEHOLDER, MENTION_PLACEHOLDER, CASHTAG_PLACEHOLDER}

# Preprocessing — financial noise tokens (used on top of NLTK stopwords)
FINANCIAL_STOPWORDS = {
    "rt", "amp", "co", "qt", "http", "https", "via",
    "stock", "stocks", "ticker", "tickers", "share", "shares",
}

# EDA — display
LABEL_PALETTE = {
    "Bearish": "#E06666",
    "Bullish": "#6AA84F",
    "Neutral": "#4A90E2",
}

# EDA — artifact detection patterns
URL_PATTERN     = re.compile(r'https?://\S+|www\.\S+')
MENTION_PATTERN = re.compile(r'@\w+')
CASHTAG_PATTERN = re.compile(r'\$\w+')
HASHTAG_PATTERN = re.compile(r'#\w+')

# EDA — fast stopword set (no NLTK dependency)
DEFAULT_STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
    "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which",
    "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by",
    "for", "with", "about", "against", "between", "into", "through", "during", "before",
    "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over",
    "under", "again", "further", "then", "once", "here", "there", "when", "where", "why",
    "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can",
    "will", "just", "don", "should", "now", "d", "ll", "m", "o", "re", "ve", "y",
    *FINANCIAL_STOPWORDS,
}
