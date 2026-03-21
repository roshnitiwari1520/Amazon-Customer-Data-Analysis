"""
app.py
──────
Streamlit web app — loads saved model from models/ folder,
provides live sentiment prediction UI.

Run: streamlit run app.py

Author : Roshni Tiwari
Project: Amazon Customer Sentiment Analysis
"""

import streamlit as st
import numpy as np
import re
import string
import os
import sys
import pickle
import nltk

nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

# Add src to path
sys.path.insert(0, 'src')

# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Amazon Review Sentiment Analyzer",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0f1117; color: #e8e8e8; }
.main-header { text-align: center; padding: 2.5rem 0 1.5rem; }
.main-title { font-size: 2.4rem; font-weight: 600; color: #ffffff; letter-spacing: -0.5px; margin-bottom: 0.4rem; }
.main-subtitle { font-size: 1rem; color: #6b7280; font-weight: 300; }
.stTextArea textarea { background: #1a1d27 !important; border: 1px solid #2d3748 !important; border-radius: 12px !important; color: #e8e8e8 !important; font-family: 'DM Sans', sans-serif !important; font-size: 15px !important; padding: 14px 16px !important; resize: none !important; }
.stTextArea textarea:focus { border-color: #4f8ef7 !important; box-shadow: 0 0 0 3px rgba(79,142,247,0.15) !important; }
.stButton > button { background: #4f8ef7 !important; color: white !important; border: none !important; border-radius: 10px !important; padding: 0.65rem 2.5rem !important; font-size: 15px !important; font-weight: 500 !important; font-family: 'DM Sans', sans-serif !important; width: 100% !important; transition: all 0.2s !important; }
.stButton > button:hover { background: #3b7de8 !important; transform: translateY(-1px) !important; }
.result-card { border-radius: 16px; padding: 1.8rem 2rem; margin: 1rem 0; text-align: center; }
.result-positive { background: linear-gradient(135deg, #0d2e1f 0%, #0f3d28 100%); border: 1px solid #1a6640; }
.result-neutral  { background: linear-gradient(135deg, #1e1e2e 0%, #252535 100%); border: 1px solid #404060; }
.result-negative { background: linear-gradient(135deg, #2e0d0d 0%, #3d1010 100%); border: 1px solid #6e2020; }
.result-emoji { font-size: 3rem; margin-bottom: 0.5rem; }
.result-label { font-size: 1.8rem; font-weight: 600; margin-bottom: 0.3rem; }
.result-positive .result-label { color: #4ade80; }
.result-neutral  .result-label { color: #94a3b8; }
.result-negative .result-label { color: #f87171; }
.result-sub { font-size: 0.9rem; color: #6b7280; }
.conf-row { margin: 0.5rem 0; }
.conf-label { display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 4px; }
.conf-track { background: #1e2030; border-radius: 6px; height: 8px; overflow: hidden; }
.conf-fill-pos { background: #4ade80; height: 100%; border-radius: 6px; }
.conf-fill-neu { background: #94a3b8; height: 100%; border-radius: 6px; }
.conf-fill-neg { background: #f87171; height: 100%; border-radius: 6px; }
.metric-row { display: flex; gap: 12px; margin: 1rem 0; }
.metric-card { flex: 1; background: #1a1d27; border: 1px solid #2d3748; border-radius: 12px; padding: 1.1rem; text-align: center; }
.metric-num { font-size: 1.6rem; font-weight: 600; color: #4f8ef7; }
.metric-label { font-size: 11px; color: #6b7280; margin-top: 3px; text-transform: uppercase; letter-spacing: 0.05em; }
.section-head { font-size: 13px; font-weight: 500; color: #6b7280; text-transform: uppercase; letter-spacing: 0.08em; margin: 1.5rem 0 0.8rem; }
.hist-row { display: flex; align-items: center; gap: 12px; padding: 10px 14px; background: #1a1d27; border-radius: 10px; margin-bottom: 6px; border: 1px solid #2d3748; }
.hist-sentiment { font-size: 12px; font-weight: 500; padding: 2px 10px; border-radius: 20px; white-space: nowrap; }
.hist-pos { background: #0d2e1f; color: #4ade80; border: 1px solid #1a6640; }
.hist-neu { background: #1e1e2e; color: #94a3b8; border: 1px solid #404060; }
.hist-neg { background: #2e0d0d; color: #f87171; border: 1px solid #6e2020; }
.hist-text { font-size: 13px; color: #9ca3af; flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0 !important; max-width: 900px; }
</style>
""", unsafe_allow_html=True)


# ─── Stopwords ─────────────────────────────────────────────────────────────────
STOP_WORDS = set(stopwords.words('english'))
STOP_WORDS -= {'not', 'no', 'never', 'nothing', 'neither', 'nor', 'none'}


def clean_text(text):
    if not isinstance(text, str): return ''
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return ' '.join([w for w in text.split() if w not in STOP_WORDS and len(w) > 2])


# ─── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    """
    Try to load saved model from models/ folder.
    If not found, train a lightweight model on the fly.
    """
    model_path = os.path.join('models', 'sentiment_model.pkl')
    vec_path   = os.path.join('models', 'tfidf_vectorizer.pkl')

    # Load saved model if it exists (after running run_pipeline.py)
    if os.path.exists(model_path) and os.path.exists(vec_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(vec_path, 'rb') as f:
            tfidf = pickle.load(f)
        return model, tfidf, "Loaded from saved model (82.02% accuracy)"

    # Fallback — train lightweight model if no saved model found
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    import numpy as np

    np.random.seed(42)
    pos = ["absolutely love product works perfectly outstanding quality",
           "best purchase fast delivery exactly described buy again",
           "exceeded expectations highly recommend brilliant product value",
           "perfect quality great value happy purchase excellent fast",
           "fantastic packaging arrived time five stars excellent wonderful",
           "works great easy setup daily use good build quality durable"]
    neu = ["product okay does says nothing special average mediocre",
           "decent quality price not best acceptable fine ordinary",
           "arrived time works expected average nothing outstanding"]
    neg = ["terrible quality broke two days waste money disappointed",
           "do not buy completely different description awful horrible",
           "very disappointed stopped working within week poor quality",
           "cheap material poor finish returned immediately defective scam"]

    texts, labels = [], []
    for _ in range(8000):
        texts.append(np.random.choice(pos) + ' ' + np.random.choice(pos)); labels.append('Positive')
    for _ in range(3000):
        texts.append(np.random.choice(neu) + ' ' + np.random.choice(neu)); labels.append('Neutral')
    for _ in range(4000):
        texts.append(np.random.choice(neg) + ' ' + np.random.choice(neg)); labels.append('Negative')

    cleaned = [clean_text(t) for t in texts]
    tfidf   = TfidfVectorizer(max_features=10000, ngram_range=(1,2), min_df=2)
    X       = tfidf.fit_transform(cleaned)
    model   = LogisticRegression(C=1.0, max_iter=500, class_weight='balanced',
                                  random_state=42, solver='lbfgs', multi_class='multinomial')
    model.fit(X, labels)
    return model, tfidf, "Lightweight model (run pipeline for full accuracy)"


# ─── Session state ─────────────────────────────────────────────────────────────
if 'history'   not in st.session_state: st.session_state.history   = []
if 'total'     not in st.session_state: st.session_state.total     = 0
if 'pos_count' not in st.session_state: st.session_state.pos_count = 0
if 'neg_count' not in st.session_state: st.session_state.neg_count = 0


# ─── Load ──────────────────────────────────────────────────────────────────────
with st.spinner('Loading sentiment model...'):
    model, tfidf, model_status = load_artifacts()


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <div class="main-title">🛒 Amazon Review Sentiment</div>
    <div class="main-subtitle">Paste any product review — get instant sentiment analysis</div>
</div>
""", unsafe_allow_html=True)


# ─── Layout ───────────────────────────────────────────────────────────────────
col1, col2 = st.columns([3, 2], gap="large")

with col1:
    st.markdown('<div class="section-head">Enter Review</div>', unsafe_allow_html=True)

    examples = {
        "😊 Positive": "This product is absolutely amazing! Best purchase I have made this year. Fast delivery and perfect quality.",
        "😐 Neutral" : "Product is okay. Does what it says but nothing really special about it. Average for the price.",
        "😞 Negative": "Terrible product. Broke after two days of normal use. Complete waste of money. Very disappointed.",
    }

    ex_cols = st.columns(3)
    selected = None
    for i, (label, text) in enumerate(examples.items()):
        if ex_cols[i].button(label, key=f'ex_{i}'):
            selected = text

    review_input = st.text_area(
        label="",
        value=selected if selected else "",
        height=160,
        placeholder="Type or paste an Amazon product review here...",
        label_visibility="collapsed"
    )

    word_count = len(review_input.split()) if review_input.strip() else 0
    st.caption(f"{word_count} words · {len(review_input)} characters")

    analyze_btn = st.button("Analyze Sentiment →", type="primary")

    if analyze_btn and review_input.strip():
        cleaned  = clean_text(review_input)
        vec      = tfidf.transform([cleaned])
        pred     = model.predict(vec)[0]
        probs    = model.predict_proba(vec)[0]
        prob_dict = dict(zip(model.classes_, probs))

        st.session_state.total += 1
        if pred == 'Positive': st.session_state.pos_count += 1
        elif pred == 'Negative': st.session_state.neg_count += 1
        st.session_state.history.insert(0, {'text': review_input[:80], 'sentiment': pred})
        if len(st.session_state.history) > 8: st.session_state.history.pop()

        card_class = f"result-{pred.lower()}"
        emoji_map  = {'Positive', 'Neutral', 'Negative'}

        st.markdown(f"""
        <div class="result-card {card_class}">
            <div class="result-emoji">{emoji_map[pred]}</div>
            <div class="result-label">{pred}</div>
            <div class="result-sub">Detected with {prob_dict[pred]*100:.1f}% confidence</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-head">Confidence Breakdown</div>', unsafe_allow_html=True)
        for sentiment, fill_class in [('Positive','pos'), ('Neutral','neu'), ('Negative','neg')]:
            pct = prob_dict.get(sentiment, 0) * 100
            color = '#4ade80' if sentiment=='Positive' else '#94a3b8' if sentiment=='Neutral' else '#f87171'
            st.markdown(f"""
            <div class="conf-row">
                <div class="conf-label">
                    <span style="color:{color}">{sentiment}</span>
                    <span style="color:#6b7280;font-family:'DM Mono',monospace">{pct:.1f}%</span>
                </div>
                <div class="conf-track">
                    <div class="conf-fill-{fill_class}" style="width:{pct}%"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    elif analyze_btn:
        st.warning("Please enter a review first.")


with col2:
    st.markdown('<div class="section-head">Session Stats</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card"><div class="metric-num">{st.session_state.total}</div><div class="metric-label">Analyzed</div></div>
        <div class="metric-card"><div class="metric-num" style="color:#4ade80">{st.session_state.pos_count}</div><div class="metric-label">Positive</div></div>
        <div class="metric-card"><div class="metric-num" style="color:#f87171">{st.session_state.neg_count}</div><div class="metric-label">Negative</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-head">Model Info</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:#1a1d27;border:1px solid #2d3748;border-radius:12px;padding:1rem 1.2rem;">
        <div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #2d3748;font-size:13px;"><span style="color:#6b7280">Dataset</span><span style="color:#e8e8e8">Amazon Fine Food Reviews</span></div>
        <div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #2d3748;font-size:13px;"><span style="color:#6b7280">Reviews</span><span style="color:#e8e8e8">100,000+</span></div>
        <div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #2d3748;font-size:13px;"><span style="color:#6b7280">Algorithm</span><span style="color:#e8e8e8">Logistic Regression</span></div>
        <div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #2d3748;font-size:13px;"><span style="color:#6b7280">Vectorizer</span><span style="color:#e8e8e8">TF-IDF (20k features)</span></div>
        <div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #2d3748;font-size:13px;"><span style="color:#6b7280">Accuracy</span><span style="color:#4ade80;font-weight:500">82.02%</span></div>
        <div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #2d3748;font-size:13px;"><span style="color:#6b7280">Classes</span><span style="color:#e8e8e8">Positive · Neutral · Negative</span></div>
        <div style="display:flex;justify-content:space-between;padding:5px 0;font-size:12px;"><span style="color:#374151">Status</span><span style="color:#374151">{model_status}</span></div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.history:
        st.markdown('<div class="section-head">Recent Analyses</div>', unsafe_allow_html=True)
        for item in st.session_state.history[:5]:
            cls = {'Positive': 'pos', 'Neutral': 'neu', 'Negative': 'neg'}[item['sentiment']]
            st.markdown(f"""
            <div class="hist-row">
                <span class="hist-sentiment hist-{cls}">{item['sentiment']}</span>
                <span class="hist-text">{item['text']}...</span>
            </div>
            """, unsafe_allow_html=True)


st.markdown("""
<div style="text-align:center;padding:2rem 0 1rem;color:#374151;font-size:12px;">
    Built by Roshni Tiwari · Amazon Sentiment Analysis · Python · NLP · Scikit-learn
</div>
""", unsafe_allow_html=True)
