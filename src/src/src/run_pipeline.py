"""
run_pipeline.py
───────────────
Master script — runs the complete pipeline end to end:

    Step 1: Preprocess  → loads Reviews.csv, cleans text, saves cleaned.csv
    Step 2: Train       → trains model, saves model + vectorizer + metrics
    Step 3: Predict     → demo predictions on sample reviews

Usage:
    python run_pipeline.py

Author : Roshni Tiwari
Project: Amazon Customer Sentiment Analysis
"""

import os
import sys
import time

# Add src/ to path so we can import from it
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from preprocess import run_preprocessing
from train      import run_training
from predict    import predict_sentiment, load_model


def print_banner():
    print("\n" + "=" * 60)
    print("   AMAZON SENTIMENT ANALYSIS — FULL PIPELINE")
    print("   Author: Roshni Tiwari")
    print("=" * 60 + "\n")


def run_full_pipeline():
    print_banner()
    start_total = time.time()

    # ── STEP 1: Preprocessing ─────────────────────────────────────────────────
    print("STAGE 1 — DATA PREPROCESSING")
    print("-" * 40)
    start = time.time()
    df = run_preprocessing()
    print(f"Time taken: {time.time() - start:.1f}s\n")

    # ── STEP 2: Training ──────────────────────────────────────────────────────
    print("STAGE 2 — MODEL TRAINING")
    print("-" * 40)
    start = time.time()
    model, tfidf = run_training()
    print(f"Time taken: {time.time() - start:.1f}s\n")

    # ── STEP 3: Demo Predictions ──────────────────────────────────────────────
    print("STAGE 3 — DEMO PREDICTIONS")
    print("-" * 40)

    demo_reviews = [
        "This product is absolutely amazing! Best purchase ever. Highly recommend.",
        "Terrible. Broke after two days. Complete waste of money. Very disappointed.",
        "Product is okay. Average quality. Nothing special about it.",
        "Delicious taste! Fresh and great packaging. Will order again for sure.",
        "Do not buy this. Completely different from the description. Total scam.",
    ]

    emoji_map = {'Positive', 'Neutral', 'Negative'}

    for review in demo_reviews:
        result = predict_sentiment(review, model, tfidf)
        emoji  = emoji_map[result['sentiment']]
        short  = review[:55] + '...' if len(review) > 55 else review
        print(f"  {emoji} {result['sentiment']:10s} ({result['confidence']:5.1f}%)  |  {short}")

    # ── Summary ───────────────────────────────────────────────────────────────
    total_time = time.time() - start_total
    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE in {total_time:.1f} seconds")
    print(f"  Model saved to  : models/sentiment_model.pkl")
    print(f"  Vectorizer saved: models/tfidf_vectorizer.pkl")
    print(f"  Metrics saved   : models/metrics.json")
    print(f"\n  Next step: streamlit run app.py")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    run_full_pipeline()
