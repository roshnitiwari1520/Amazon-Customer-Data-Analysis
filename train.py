"""
train.py
────────
Step 2 of the pipeline — Load cleaned data, vectorize with TF-IDF,
train Logistic Regression, evaluate, save model + vectorizer.

Author : Roshni Tiwari
Project: Amazon Customer Sentiment Analysis
"""

import pandas as pd
import numpy as np
import pickle
import os
import json
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ── Constants ──────────────────────────────────────────────────────────────────
CLEANED_DATA_PATH = os.path.join('data',   'cleaned.csv')
MODEL_PATH        = os.path.join('models', 'sentiment_model.pkl')
VECTORIZER_PATH   = os.path.join('models', 'tfidf_vectorizer.pkl')
METRICS_PATH      = os.path.join('models', 'metrics.json')
RANDOM_STATE      = 42
TEST_SIZE         = 0.20


# ── Functions ──────────────────────────────────────────────────────────────────

def load_cleaned_data(path: str) -> pd.DataFrame:
    """Load the cleaned CSV produced by preprocess.py"""
    print(f"[1/5] Loading cleaned data from: {path}")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cleaned data not found at '{path}'.\n"
            f"Run preprocess.py first: python src/preprocess.py"
        )

    df = pd.read_csv(path)
    df = df.dropna(subset=['clean_text', 'sentiment'])
    df = df[df['clean_text'].str.strip() != ''].reset_index(drop=True)

    print(f"    Loaded {len(df):,} cleaned reviews")
    print(f"    Classes: {df['sentiment'].value_counts().to_dict()}")
    return df


def vectorize(X_train, X_test):
    """
    Convert text to TF-IDF numeric vectors.

    Why TF-IDF?
    - Gives higher weight to words important in a review but rare across all reviews
    - 'defective' scores high in a bad review; 'the' scores low everywhere
    - ngram_range=(1,2) captures bigrams like 'not good', 'highly recommend'
    """
    print("[3/5] Fitting TF-IDF vectorizer...")

    tfidf = TfidfVectorizer(
        max_features=20000,     # top 20,000 words/phrases
        ngram_range=(1, 2),     # unigrams + bigrams
        min_df=3,               # ignore words in fewer than 3 reviews
        sublinear_tf=True       # log scaling on term frequency
    )

    # Fit ONLY on training data — fitting on test = data leakage
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec  = tfidf.transform(X_test)

    print(f"    Vocabulary size : {len(tfidf.vocabulary_):,} features")
    print(f"    Train matrix    : {X_train_vec.shape[0]:,} x {X_train_vec.shape[1]:,}")

    return tfidf, X_train_vec, X_test_vec


def train_model(X_train_vec, y_train):
    """
    Train Logistic Regression classifier.

    Why Logistic Regression?
    - Handles high-dimensional sparse matrices well
    - Fast even on 100k reviews
    - Interpretable — coefficients show which words drive sentiment
    - Strong NLP baseline before trying BERT/transformers
    """
    print("[4/5] Training Logistic Regression model...")

    model = LogisticRegression(
        C=1.0,                       # regularization strength
        max_iter=1000,               # max iterations to converge
        class_weight='balanced',     # handles class imbalance automatically
        random_state=RANDOM_STATE,
        solver='lbfgs',
        multi_class='multinomial'    # 3 output classes
    )

    model.fit(X_train_vec, y_train)
    print("    Training complete!")
    return model


def evaluate_model(model, X_test_vec, y_test):
    """Evaluate model and return metrics dictionary"""
    print("[5/5] Evaluating model...")

    y_pred   = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    report   = classification_report(
        y_test, y_pred,
        target_names=['Negative', 'Neutral', 'Positive'],
        output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred,
                          labels=['Negative', 'Neutral', 'Positive'])

    print(f"\n{'='*50}")
    print("    MODEL EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"    Overall Accuracy  : {accuracy*100:.2f}%")
    print(f"\n    Per-class results:")
    for cls in ['Negative', 'Neutral', 'Positive']:
        r = report.get(cls, {})
        print(f"    {cls:10s}  precision={r.get('precision',0):.2f}  "
              f"recall={r.get('recall',0):.2f}  f1={r.get('f1-score',0):.2f}")
    print(f"{'='*50}\n")

    metrics = {
        'accuracy'        : round(accuracy, 4),
        'trained_at'      : datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'train_size'      : int(len(y_test) / TEST_SIZE * (1 - TEST_SIZE)),
        'test_size'       : int(len(y_test)),
        'classification_report': report
    }
    return metrics, y_pred


def save_artifacts(model, tfidf, metrics):
    """Save model, vectorizer and metrics to models/ folder"""
    os.makedirs('models', exist_ok=True)

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    print(f"    Model saved     : {MODEL_PATH}")

    with open(VECTORIZER_PATH, 'wb') as f:
        pickle.dump(tfidf, f)
    print(f"    Vectorizer saved: {VECTORIZER_PATH}")

    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"    Metrics saved   : {METRICS_PATH}")


def run_training() -> tuple:
    """
    Full training pipeline:
    Load → Split → Vectorize → Train → Evaluate → Save
    Returns (model, tfidf_vectorizer)
    """
    # Step 1: Load cleaned data
    df = load_cleaned_data(CLEANED_DATA_PATH)

    # Step 2: Train-test split
    print(f"[2/5] Splitting data ({int((1-TEST_SIZE)*100)}% train / {int(TEST_SIZE*100)}% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'],
        df['sentiment'],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df['sentiment']    # maintain class ratio in both splits
    )
    print(f"    Train: {len(X_train):,}  Test: {len(X_test):,}")

    # Step 3: Vectorize
    tfidf, X_train_vec, X_test_vec = vectorize(X_train, X_test)

    # Step 4: Train
    model = train_model(X_train_vec, y_train)

    # Step 5: Evaluate + Save
    metrics, _ = evaluate_model(model, X_test_vec, y_test)
    save_artifacts(model, tfidf, metrics)

    print("Pipeline complete! Model ready.\n")
    return model, tfidf


# ── Run directly ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    run_training()
