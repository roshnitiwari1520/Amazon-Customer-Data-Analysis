"""
predict.py
──────────
Step 3 of the pipeline — Load saved model + vectorizer,
predict sentiment on any new review text.

Author : Roshni Tiwari
Project: Amazon Customer Sentiment Analysis
"""

import pickle
import re
import string
import os
import nltk

nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords


# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_PATH      = os.path.join('models', 'sentiment_model.pkl')
VECTORIZER_PATH = os.path.join('models', 'tfidf_vectorizer.pkl')

# ── Stopwords ─────────────────────────────────────────────────────────────────
STOP_WORDS = set(stopwords.words('english'))
KEEP_WORDS = {'not', 'no', 'never', 'nothing', 'neither', 'nor', 'none'}
STOP_WORDS = STOP_WORDS - KEEP_WORDS


# ── Functions ──────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Same cleaning function used during training — must be identical"""
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = [w for w in text.split()
               if w not in STOP_WORDS and len(w) > 2]
    return ' '.join(tokens)


def load_model():
    """Load saved model and vectorizer from models/ folder"""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at '{MODEL_PATH}'.\n"
            f"Run training first: python src/train.py"
        )
    if not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(
            f"Vectorizer not found at '{VECTORIZER_PATH}'.\n"
            f"Run training first: python src/train.py"
        )

    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, 'rb') as f:
        tfidf = pickle.load(f)

    return model, tfidf


def predict_sentiment(text: str, model=None, tfidf=None) -> dict:
    """
    Predict sentiment of a single review string.

    Args:
        text  : raw review text
        model : loaded model (loads from file if None)
        tfidf : loaded vectorizer (loads from file if None)

    Returns:
        dict with keys: sentiment, confidence, all_scores
    """
    if model is None or tfidf is None:
        model, tfidf = load_model()

    cleaned    = clean_text(text)
    vec        = tfidf.transform([cleaned])
    prediction = model.predict(vec)[0]
    probs      = model.predict_proba(vec)[0]
    classes    = model.classes_

    prob_dict = dict(zip(classes, probs))

    return {
        'sentiment'  : prediction,
        'confidence' : round(prob_dict[prediction] * 100, 2),
        'all_scores' : {k: round(v * 100, 2) for k, v in prob_dict.items()}
    }


def predict_batch(texts: list, model=None, tfidf=None) -> list:
    """
    Predict sentiment for a list of reviews.

    Args:
        texts : list of review strings

    Returns:
        list of result dicts
    """
    if model is None or tfidf is None:
        model, tfidf = load_model()

    results = []
    for text in texts:
        result = predict_sentiment(text, model, tfidf)
        results.append(result)
    return results


# ── Run directly — demo predictions ───────────────────────────────────────────
if __name__ == '__main__':
    print("Loading model...")
    model, tfidf = load_model()
    print("Model loaded!\n")

    test_reviews = [
        "This product is absolutely amazing! Best purchase I have made. Fast delivery.",
        "Terrible product. Broke after two days. Complete waste of money.",
        "Product is okay. Does what it says but nothing really special.",
        "Not bad but not great. Average quality for the price paid.",
        "Highly recommend! Excellent quality and arrived right on time.",
    ]

    print("=" * 60)
    print("LIVE SENTIMENT PREDICTIONS")
    print("=" * 60)

    emoji_map = {'Positive': '😊', 'Neutral': '😐', 'Negative': '😞'}

    for review in test_reviews:
        result = predict_sentiment(review, model, tfidf)
        emoji  = emoji_map[result['sentiment']]
        print(f"\nReview    : {review[:70]}...")
        print(f"Sentiment : {emoji} {result['sentiment']}")
        print(f"Confidence: {result['confidence']}%")
        print(f"All scores: {result['all_scores']}")

    print("\n" + "=" * 60)
