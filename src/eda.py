#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
src/eda.py

Exploratory Data Analysis (EDA) Tool for Finance Tweets.
Designed as a modular library and an interactive command-line tool.
Can be executed seamlessly by AI agents or human developers to analyze,
visualize, and summarize dataset properties.
"""

import os
import re
import sys
import argparse
import json
from collections import Counter
import pandas as pd

# Try to import plotting libraries gracefully
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

# Try to import wordcloud gracefully
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

# --- Default Configs ---
LABEL_NAMES = {0: "Bearish", 1: "Bullish", 2: "Neutral"}
LABEL_PALETTE = {
    "Bearish": "#E06666",  # Harmonic red
    "Bullish": "#6AA84F",  # Harmonic green
    "Neutral": "#4A90E2"   # Harmonic blue
}

# Regex Patterns for Analysis
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
MENTION_PATTERN = re.compile(r'@\w+')
CASHTAG_PATTERN = re.compile(r'\$\w+')
HASHTAG_PATTERN = re.compile(r'#\w+')

# Simple stopword list to use if NLTK is not fully loaded or for fast baseline
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
    # Financial Noise
    "rt", "amp", "co", "qt", "http", "https", "via", "stock", "stocks", "ticker", 
    "tickers", "share", "shares"
}

# --- Core Analyzer Class ---

class DatasetAnalyzer:
    def __init__(self, file_path: str, name: str = "Dataset"):
        self.file_path = file_path
        self.name = name
        self.df = None
        self.has_label = False
        self.load_data()

    def load_data(self):
        """Loads dataset from CSV and determines if labels exist."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Dataset not found at: {self.file_path}")
        self.df = pd.read_csv(self.file_path)
        self.has_label = "label" in self.df.columns
        # Basic text cleaning to ensure str type
        self.df["text"] = self.df["text"].fillna("").astype(str)

    def analyze_basic_stats(self) -> dict:
        """Returns standard metrics on size, duplicates, and missing rows."""
        total_rows = len(self.df)
        exact_dups = self.df["text"].duplicated().sum()
        ci_dups = self.df["text"].str.lower().str.strip().duplicated().sum()
        empty_rows = (self.df["text"].str.strip() == "").sum()
        
        return {
            "total_rows": total_rows,
            "exact_duplicates": int(exact_dups),
            "trimmed_case_insensitive_duplicates": int(ci_dups),
            "empty_rows": int(empty_rows)
        }

    def analyze_class_distribution(self) -> list:
        """Returns class distribution metrics (counts, percentages) if labels are present."""
        if not self.has_label:
            return []
        
        counts = self.df["label"].value_counts().sort_index()
        total = counts.sum()
        dist = []
        for val, cnt in counts.items():
            dist.append({
                "label_id": int(val),
                "label_name": LABEL_NAMES.get(val, f"Unknown ({val})"),
                "count": int(cnt),
                "percentage": float((cnt / total * 100))
            })
        return dist

    def analyze_text_lengths(self) -> dict:
        """Computes text character length and token count statistics."""
        char_lens = self.df["text"].str.len()
        token_lens = self.df["text"].str.split().str.len()
        
        stats = {
            "char_len": {
                "mean": float(char_lens.mean()),
                "median": float(char_lens.median()),
                "std": float(char_lens.std()),
                "min": int(char_lens.min()),
                "max": int(char_lens.max())
            },
            "token_len": {
                "mean": float(token_lens.mean()),
                "median": float(token_lens.median()),
                "std": float(token_lens.std()),
                "min": int(token_lens.min()),
                "max": int(token_lens.max())
            }
        }
        
        # Breakdown by class if available
        if self.has_label:
            stats["by_class"] = {}
            for val, name in LABEL_NAMES.items():
                class_df = self.df[self.df["label"] == val]
                if not class_df.empty:
                    c_chars = class_df["text"].str.len()
                    c_tokens = class_df["text"].str.split().str.len()
                    stats["by_class"][name] = {
                        "char_mean": float(c_chars.mean()),
                        "char_median": float(c_chars.median()),
                        "token_mean": float(c_tokens.mean()),
                        "token_median": float(c_tokens.median())
                    }
        return stats

    def analyze_artifacts(self) -> dict:
        """Counts URLs, mentions, cashtags, and hashtags in the corpus."""
        def count_pat(text, pattern):
            return len(pattern.findall(text))
            
        n_urls = self.df["text"].apply(lambda t: count_pat(t, URL_PATTERN))
        n_mentions = self.df["text"].apply(lambda t: count_pat(t, MENTION_PATTERN))
        n_cashtags = self.df["text"].apply(lambda t: count_pat(t, CASHTAG_PATTERN))
        n_hashtags = self.df["text"].apply(lambda t: count_pat(t, HASHTAG_PATTERN))
        
        total = len(self.df)
        rates = {
            "average_per_tweet": {
                "urls": float(n_urls.mean()),
                "mentions": float(n_mentions.mean()),
                "cashtags": float(n_cashtags.mean()),
                "hashtags": float(n_hashtags.mean())
            },
            "presence_percentage": {
                "urls": float((n_urls > 0).sum() / total * 100),
                "mentions": float((n_mentions > 0).sum() / total * 100),
                "cashtags": float((n_cashtags > 0).sum() / total * 100),
                "hashtags": float((n_hashtags > 0).sum() / total * 100)
            }
        }

        # Breakdown by class if available
        if self.has_label:
            rates["by_class"] = {}
            for val, name in LABEL_NAMES.items():
                class_df = self.df[self.df["label"] == val]
                if not class_df.empty:
                    c_total = len(class_df)
                    rates["by_class"][name] = {
                        "urls_mean": float(class_df["text"].apply(lambda t: count_pat(t, URL_PATTERN)).mean()),
                        "mentions_mean": float(class_df["text"].apply(lambda t: count_pat(t, MENTION_PATTERN)).mean()),
                        "cashtags_mean": float(class_df["text"].apply(lambda t: count_pat(t, CASHTAG_PATTERN)).mean()),
                        "hashtags_mean": float(class_df["text"].apply(lambda t: count_pat(t, HASHTAG_PATTERN)).mean())
                    }
        return rates

    def analyze_non_ascii(self) -> dict:
        """Finds non-ASCII character rates and compiles some raw examples."""
        def has_non_ascii(text):
            return any(ord(c) > 127 for c in text)
            
        non_ascii_mask = self.df["text"].apply(has_non_ascii)
        n_non_ascii = non_ascii_mask.sum()
        total = len(self.df)
        
        # Get up to 5 examples
        examples = self.df[non_ascii_mask]["text"].head(5).tolist()
        
        return {
            "non_ascii_count": int(n_non_ascii),
            "non_ascii_percentage": float(n_non_ascii / total * 100),
            "examples": examples
        }

    def get_top_tokens(self, top_n: int = 15) -> dict:
        """Computes top N words across the dataset, excluding standard and financial stopwords."""
        # Simple token clean function
        def clean_split(text):
            # Strip URLs, mentions, cashtags first to get real words
            t = URL_PATTERN.sub('', text)
            t = MENTION_PATTERN.sub('', t)
            t = CASHTAG_PATTERN.sub('', t)
            t = HASHTAG_PATTERN.sub('', t)
            # Remove punctuation
            t = re.sub(r'[^a-zA-Z\s]', '', t)
            words = t.lower().split()
            return [w for w in words if w not in DEFAULT_STOPWORDS and len(w) > 2]

        top_tokens = {}
        
        # Global top words
        all_words = []
        for text in self.df["text"]:
            all_words.extend(clean_split(text))
        
        top_tokens["global"] = [{"word": w, "count": c} for w, c in Counter(all_words).most_common(top_n)]
        
        # Class specific top words
        if self.has_label:
            top_tokens["by_class"] = {}
            for val, name in LABEL_NAMES.items():
                class_df = self.df[self.df["label"] == val]
                c_words = []
                for text in class_df["text"]:
                    c_words.extend(clean_split(text))
                top_tokens["by_class"][name] = [{"word": w, "count": c} for w, c in Counter(c_words).most_common(top_n)]
                
        return top_tokens

    def generate_full_report(self, top_n_words: int = 15) -> dict:
        """Compiles all individual analytical components into a unified structured report."""
        return {
            "dataset_name": self.name,
            "file_path": self.file_path,
            "basic_stats": self.analyze_basic_stats(),
            "class_distribution": self.analyze_class_distribution(),
            "text_lengths": self.analyze_text_lengths(),
            "artifacts": self.analyze_artifacts(),
            "non_ascii": self.analyze_non_ascii(),
            "top_words": self.get_top_tokens(top_n=top_n_words)
        }

# --- Plot Generation Utility ---

def save_eda_plots(train_analyzer: DatasetAnalyzer, output_dir: str = "outputs/eda/"):
    """
    Generates and saves professional analytical figures under outputs/eda/.
    Requires matplotlib and seaborn. Skips gracefully if unavailable.
    """
    if not PLOTTING_AVAILABLE:
        print("[WARNING] Plotting packages (matplotlib/seaborn) not available. Skipping chart generation.")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    df = train_analyzer.df
    
    # Set tailored aesthetic style
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.size': 10, 'figure.titlesize': 13})

    # Plot 1: Class Distribution
    if train_analyzer.has_label:
        dist = train_analyzer.analyze_class_distribution()
        dist_df = pd.DataFrame(dist)
        
        plt.figure(figsize=(6, 4))
        ax = sns.barplot(
            data=dist_df, x="label_name", y="count",
            palette=[LABEL_PALETTE[name] for name in dist_df["label_name"]],
            hue="label_name", legend=False
        )
        # Add labels
        for idx, row in dist_df.iterrows():
            ax.text(idx, row["count"] + 50, f"{row['count']}\n({row['percentage']:.2f}%)", ha="center")
            
        plt.title("Class Label Distribution (Training Set)", pad=15)
        plt.xlabel("Sentiment Class")
        plt.ylabel("Number of Tweets")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "class_distribution.png"), dpi=150)
        plt.close()

    # Plot 2: Length Distributions
    df["char_len"] = df["text"].str.len()
    df["token_len"] = df["text"].str.split().str.len()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    if train_analyzer.has_label:
        plot_df = df.assign(class_name=df["label"].map(LABEL_NAMES))
        sns.histplot(data=plot_df, x="char_len", hue="class_name", multiple="layer", bins=30, palette=LABEL_PALETTE, ax=axes[0])
        sns.histplot(data=plot_df, x="token_len", hue="class_name", multiple="layer", bins=20, palette=LABEL_PALETTE, ax=axes[1])
    else:
        sns.histplot(data=df, x="char_len", bins=30, color="#4A90E2", ax=axes[0])
        sns.histplot(data=df, x="token_len", bins=20, color="#4A90E2", ax=axes[1])
        
    axes[0].set_title("Character Length Distribution")
    axes[0].set_xlabel("Number of Characters")
    axes[1].set_title("Word (Token) Length Distribution")
    axes[1].set_xlabel("Number of Words")
    
    plt.suptitle("Tweet Length Distributions", y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "length_distributions.png"), dpi=150)
    plt.close()

    # Plot 3: Artifact Rates by Class
    if train_analyzer.has_label:
        rates = train_analyzer.analyze_artifacts()
        by_class = rates.get("by_class", {})
        
        classes = list(by_class.keys())
        features = ["URLs", "Cashtags", "Hashtags", "Mentions"]
        
        plot_data = []
        for cls in classes:
            plot_data.append({
                "Class": cls,
                "URLs": by_class[cls]["urls_mean"],
                "Cashtags": by_class[cls]["cashtags_mean"],
                "Hashtags": by_class[cls]["hashtags_mean"],
                "Mentions": by_class[cls]["mentions_mean"]
            })
        
        rates_df = pd.DataFrame(plot_data).melt(id_vars="Class", var_name="Artifact", value_name="Average Count")
        
        plt.figure(figsize=(9, 4.5))
        sns.barplot(data=rates_df, x="Artifact", y="Average Count", hue="Class", palette=LABEL_PALETTE)
        plt.title("Average Metadata Artifact Rates Grouped by Sentiment Class", pad=15)
        plt.xlabel("Tweet Metadata Type")
        plt.ylabel("Average Count per Tweet")
        plt.legend(title="Sentiment Class")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "artefact_rates_by_class.png"), dpi=150)
        plt.close()

    # Plot 4: Top Words by Class
    if train_analyzer.has_label:
        top_words = train_analyzer.get_top_tokens(top_n=10)
        by_class = top_words.get("by_class", {})
        
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        for ax, (cls_name, words) in zip(axes, by_class.items()):
            words_df = pd.DataFrame(words)
            if not words_df.empty:
                sns.barplot(data=words_df, x="count", y="word", color=LABEL_PALETTE[cls_name], ax=ax)
            ax.set_title(f"Top Words in {cls_name} Tweets")
            ax.set_xlabel("Occurrence Frequency")
            ax.set_ylabel("")
            
        plt.suptitle("Top Highly Informative Sentiment Vocabularies", y=0.98)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "top_words_by_class.png"), dpi=150)
        plt.close()
        
    print(f"[SUCCESS] Visual charts saved perfectly under {output_dir}")

# --- Output Formatters ---

def print_markdown_report(train_rpt: dict, test_rpt: dict = None):
    """Prints a beautiful, comprehensive Markdown formatted report to stdout."""
    print(f"# Exploratory Data Analysis (EDA) Report")
    print(f"**Generative Analysis for Corpus Pipelines**\n")
    
    # 1. Dataset Overview
    print("## [1] Dataset Size & Integrity Overview")
    print("| Dataset | Total Rows | Exact Duplicates | Trimmed Insensitive Dups | Empty Rows |")
    print("| :--- | :---: | :---: | :---: | :---: |")
    tr = train_rpt["basic_stats"]
    print(f"| **Train** | {tr['total_rows']:,} | {tr['exact_duplicates']} | {tr['trimmed_case_insensitive_duplicates']} | {tr['empty_rows']} |")
    if test_rpt:
        te = test_rpt["basic_stats"]
        print(f"| **Test** | {te['total_rows']:,} | {te['exact_duplicates']} | {te['trimmed_case_insensitive_duplicates']} | {te['empty_rows']} |")
    print()

    # 2. Class Distribution
    if train_rpt["class_distribution"]:
        print("## [2] Sentiment Class Distributions (Train Set)")
        print("| Label ID | Sentiment Class | Count | Percentage | Chart representation |")
        print("| :---: | :--- | :---: | :---: | :--- |")
        for row in train_rpt["class_distribution"]:
            bar = "=" * int(row["percentage"] / 5)
            print(f"| `{row['label_id']}` | **{row['label_name']}** | {row['count']:,} | {row['percentage']:.2f}% | `{bar}` |")
        print("\n> **Strategic Note**: Standard ML algorithms will show bias to the majority Neutral class. Set class weight parameters to 'balanced' or implement cross-validation splitting accordingly.\n")

    # 3. Tweet Length Statistics
    print("## [3] Sequence Length Analyses")
    l_tr = train_rpt["text_lengths"]
    print("### Character Lengths:")
    print(f"* **Train**: Mean = {l_tr['char_len']['mean']:.1f} chars, Std = {l_tr['char_len']['std']:.1f}, Min/Max = {l_tr['char_len']['min']}/{l_tr['char_len']['max']}")
    if test_rpt:
        l_te = test_rpt["text_lengths"]
        print(f"* **Test**: Mean = {l_te['char_len']['mean']:.1f} chars, Std = {l_te['char_len']['std']:.1f}, Min/Max = {l_te['char_len']['min']}/{l_te['char_len']['max']}")
    
    print("\n### Word (Token) Counts:")
    print(f"* **Train**: Mean = {l_tr['token_len']['mean']:.1f} words, Std = {l_tr['token_len']['std']:.1f}, Min/Max = {l_tr['token_len']['min']}/{l_tr['token_len']['max']}")
    if test_rpt:
        print(f"* **Test**: Mean = {l_te['token_len']['mean']:.1f} words, Std = {l_te['token_len']['std']:.1f}, Min/Max = {l_te['token_len']['min']}/{l_te['token_len']['max']}")
    
    if "by_class" in l_tr:
        print("\n### Sequence Characteristics Grouped by Class (Train):")
        print("| Sentiment Class | Avg Character Length | Avg Word (Token) Count |")
        print("| :--- | :---: | :---: |")
        for cls_name, vals in l_tr["by_class"].items():
            print(f"| **{cls_name}** | {vals['char_mean']:.1f} characters | {vals['token_mean']:.1f} tokens |")
    print()

    # 4. Artifact Analysis
    print("## [4] Twitter Metadata Artifact Rates")
    art = train_rpt["artifacts"]
    print("| Metadata Type | Average Occurrences / Tweet | Percentage of Tweets Containing Feature |")
    print("| :--- | :---: | :---: |")
    for feat in ["urls", "cashtags", "hashtags", "mentions"]:
        print(f"| **{feat.upper()}** | {art['average_per_tweet'][feat]:.3f} | {art['presence_percentage'][feat]:.1f}% |")
    
    if "by_class" in art:
        print("\n### Average Metadata Artifact Occurrences Grouped by Class (Train):")
        print("| Sentiment Class | Avg URLs / Tweet | Avg Cashtags / Tweet | Avg Hashtags / Tweet | Avg Mentions / Tweet |")
        print("| :--- | :---: | :---: | :---: | :---: |")
        for cls_name, vals in art["by_class"].items():
            print(f"| **{cls_name}** | {vals['urls_mean']:.3f} | {vals['cashtags_mean']:.3f} | {vals['hashtags_mean']:.3f} | {vals['mentions_mean']:.3f} |")
    print("\n> **Strategic Note**: URLs are highly associated with Neutral (news) tweets. Removing URLs completely strips this structural feature; utilizing a unified placeholder `URL_PLACEHOLDER` is highly recommended.\n")

    # 5. Non-ASCII Analysis
    print("## [5] Non-ASCII & Unicode Anomalies")
    non_tr = train_rpt["non_ascii"]
    print(f"* **Train**: {non_tr['non_ascii_count']:,} tweets ({non_tr['non_ascii_percentage']:.2f}%) contain smart quotes, long dashes, smart apostrophes, or replacement symbols.")
    if test_rpt:
        non_te = test_rpt["non_ascii"]
        print(f"* **Test**: {non_te['non_ascii_count']:,} tweets ({non_te['non_ascii_percentage']:.2f}%) contain smart quotes, long dashes, smart apostrophes, or replacement symbols.")
    
    print("\n### Sample raw sentences showing Unicode irregularities:")
    for ex in non_tr["examples"]:
        # Clean sample sentences of emojis in case they cause issues when printed in samples
        clean_ex = re.sub(r'[^\x00-\x7F]+', ' ', ex)
        print(f"* `{clean_ex}`")
    print()

    # 6. Vocabularies
    print("## [6] Most Frequent Sentiment Vocabularies (Excluding Noise)")
    top_w = train_rpt["top_words"]
    print("### Top Overall Vocabularies:")
    print(", ".join([f"**{row['word']}** ({row['count']})" for row in top_w["global"]]))
    
    if "by_class" in top_w:
        print("\n### Top Vocabularies by Class (Train):")
        for cls_name, words in top_w["by_class"].items():
            word_list = ", ".join([f"**{row['word']}** ({row['count']})" for row in words[:8]])
            print(f"* **{cls_name}**: {word_list}")
    print()

# --- Main Entry Point ---

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Comprehensive EDA Analyzer for Text Mining Finance Tweets. Built as a tool to automate analysis."
    )
    parser.add_argument('--train-path', type=str, default='data/train.csv', help="Path to training set CSV file.")
    parser.add_argument('--test-path', type=str, default='data/test.csv', help="Path to testing set CSV file.")
    parser.add_argument('--format', type=str, choices=['text', 'markdown', 'json'], default='markdown',
                        help="Format of the stdout console report.")
    parser.add_argument('--save-plots', action='store_true', help="Generate and save analysis figures under outputs/eda/.")
    parser.add_argument('--top-n', type=int, default=15, help="Number of top words to extract in analysis.")

    args = parser.parse_args()

    try:
        # Load and analyze training set
        train_analyzer = DatasetAnalyzer(args.train_path, name="Training Set")
        train_report = train_analyzer.generate_full_report(top_n_words=args.top_n)
        
        # Load and analyze testing set
        test_report = None
        if os.path.exists(args.test_path):
            test_analyzer = DatasetAnalyzer(args.test_path, name="Testing Set")
            test_report = test_analyzer.generate_full_report(top_n_words=args.top_n)

        # Output in selected format
        if args.format == 'json':
            full_json = {"train": train_report}
            if test_report:
                full_json["test"] = test_report
            print(json.dumps(full_json, indent=2))
        elif args.format == 'markdown':
            print_markdown_report(train_report, test_report)
        else:
            # Simple text output fallback
            print(f"=== EDA Overview for {args.train_path} ===")
            print(f"Total Rows: {train_report['basic_stats']['total_rows']}")
            print(f"Non-ASCII rate: {train_report['non_ascii']['non_ascii_percentage']:.2f}%")
            if test_report:
                print(f"=== EDA Overview for {args.test_path} ===")
                print(f"Total Rows: {test_report['basic_stats']['total_rows']}")

        # Save charts if requested
        if args.save_plots:
            save_eda_plots(train_analyzer)

    except Exception as e:
        if args.format == 'json':
            print(json.dumps({"status": "error", "message": str(e)}))
        else:
            print(f"Error during EDA execution: {str(e)}", file=sys.stderr)
        sys.exit(1)
