"""
preprocess.py
─────────────
Step 1 of the pipeline — Load raw CSV, clean text, label sentiment,
save cleaned data to data/cleaned.csv

Author : Roshni Tiwari
Project: Amazon Customer Sentiment Analysis
"""

import pandas as pd
import re
import string
import os
import nltk

nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords


# ── Constants ──────────────────────────────────────────────────────────────────
RAW_DATA_PATH     = os.path.join('data', 'Reviews.csv')
CLEANED_DATA_PATH = os.path.join('data', 'cleaned.csv')
SAMPLE_SIZE       = 100000
RANDOM_STATE      = 42


# ── Stopwords — keep negation words ───────────────────────────────────────────
STOP_WORDS = set(stopwords.words('english'))
KEEP_WORDS = {'not', 'no', 'never', 'nothing', 'neither', 'nor', 'none'}
STOP_WORDS = STOP_WORDS - KEEP_WORDS   # negation words must stay


# ── Functions ──────────────────────────────────────────────────────────────────

def load_raw_data(path: str, sample_size: int = SAMPLE_SIZE) -> pd.DataFrame:
    """
    Load raw Reviews.csv from Kaggle.
    Keeps only Text (review) and Score (star rating).
    Samples to sample_size rows for faster processing.
    """
    print(f"[1/4] Loading raw data from: {path}")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Reviews.csv not found at '{path}'.\n"
            f"Download from: https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews\n"
            f"Place it in the 'data/' folder."
        )

    df = pd.read_csv(path, usecols=['Text', 'Score'])
    df.columns = ['review_text', 'rating']
    df = df.dropna(subset=['review_text']).reset_index(drop=True)

    # Sample if dataset is larger than sample_size
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=RANDOM_STATE)
        df = df.reset_index(drop=True)

    print(f"    Loaded {len(df):,} reviews")
    return df


def assign_sentiment(rating: int) -> str:
    """
    Convert numeric star rating to sentiment label.
    4-5 stars → Positive
    3 stars   → Neutral
    1-2 stars → Negative
    """
    if rating >= 4:
        return 'Positive'
    elif rating == 3:
        return 'Neutral'
    else:
        return 'Negative'


def clean_text(text: str) -> str:
    """
    Clean a single review string.
    Steps: lowercase → remove URLs → remove numbers →
           remove punctuation → remove stopwords → strip short words
    """
    if not isinstance(text, str):
        return ''

    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)                  # remove URLs
    text = re.sub(r'\d+', '', text)                             # remove numbers
    text = text.translate(
        str.maketrans('', '', string.punctuation)               # remove punctuation
    )
    tokens = [
        w for w in text.split()
        if w not in STOP_WORDS and len(w) > 2                  # remove stopwords
    ]
    return ' '.join(tokens)


def run_preprocessing() -> pd.DataFrame:
    """
    Full preprocessing pipeline:
    Load → Label → Clean → Save
    Returns cleaned DataFrame.
    """
    # Step 1: Load
    df = load_raw_data(RAW_DATA_PATH)

    # Step 2: Label sentiment
    print("[2/4] Labelling sentiment from star ratings...")
    df['sentiment'] = df['rating'].apply(assign_sentiment)

    counts = df['sentiment'].value_counts()
    for label, count in counts.items():
        print(f"    {label:10s}: {count:>8,}  ({count/len(df)*100:.1f}%)")

    # Step 3: Clean text
    print("[3/4] Cleaning review text... (takes ~30 seconds)")
    df['clean_text'] = df['review_text'].apply(clean_text)

    # Remove rows where cleaning returned empty string
    before = len(df)
    df = df[df['clean_text'].str.strip() != ''].reset_index(drop=True)
    removed = before - len(df)
    if removed > 0:
        print(f"    Removed {removed} empty rows after cleaning")

    # Step 4: Save cleaned data
    print(f"[4/4] Saving cleaned data to: {CLEANED_DATA_PATH}")
    os.makedirs('data', exist_ok=True)
    df[['review_text', 'rating', 'sentiment', 'clean_text']].to_csv(
        CLEANED_DATA_PATH, index=False
    )
    print(f"    Saved {len(df):,} cleaned reviews")
    print("    Preprocessing complete!\n")

    return df


# ── Run directly ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    run_preprocessing()
