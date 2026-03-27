#  Amazon Customer Data Analysis

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat&logo=python)
![NLP](https://img.shields.io/badge/NLP-NLTK-green?style=flat)
![ML](https://img.shields.io/badge/ML-Scikit--Learn-orange?style=flat&logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/App-Streamlit-red?style=flat&logo=streamlit)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=flat)

> **Sentiment classification of Amazon product reviews using NLP and Logistic Regression**  
> Analyzed 100,000+ reviews to classify sentiment (positive/neutral/negative), extract insights, and identify top-performing products.

---

## Project Overview

This project performs end-to-end sentiment analysis on Amazon product reviews:
- Cleaned and preprocessed raw review text using NLP techniques
- Built a TF-IDF + Logistic Regression classification model
- Achieved **85%+ accuracy** on sentiment classification
- Identified top products based on sentiment scores
- Deployed an interactive **Streamlit web app** for live predictions

---

##  Dataset

This project uses the **Amazon Fine Food Reviews** dataset from Kaggle.

🔗 [Download Dataset from Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)

### Setup Instructions
1. Download the dataset from the link above
2. Create a `data/` folder in the project root
3. Place the downloaded `Reviews.csv` file inside `data/`

---

##  Project Structure

```
Amazon-Customer-Data-Analysis/
│
├── data/                    # Dataset folder (download from Kaggle)
│   └── Reviews.csv
│
├── notebooks/               # Jupyter notebooks for EDA
│   └── Amazon_Customer_Analysis.ipynb
│
├── models/                  # Saved trained models
│   └── sentiment_model.pkl
│
├── screenshots/             # App screenshots
│   └── app_demo.png
│
├── preprocess.py            # Text cleaning & preprocessing
├── train.py                 # Model training pipeline
├── predict.py               # Load model & predict sentiment
├── run_pipeline.py          # Master script — runs full pipeline
├── app.py                   # Streamlit web application
├── Requirements.txt         # Project dependencies
└── README.md
```

---

##  Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.8+ |
| NLP | NLTK, TF-IDF Vectorizer |
| Machine Learning | Scikit-Learn, Logistic Regression, Naive Bayes |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Web App | Streamlit |
| Version Control | Git, GitHub |

---

## Pipeline

```
Raw Reviews → Preprocessing → TF-IDF Features → Model Training → Prediction → Streamlit App
```

**Step-by-step:**
1. **Preprocess** — Lowercasing, punctuation removal, stopword removal, lemmatization
2. **Vectorize** — Convert text to TF-IDF features (5000 features, unigrams + bigrams)
3. **Train** — Logistic Regression classifier with stratified train/test split (80/20)
4. **Evaluate** — Classification report, confusion matrix, feature importance
5. **Deploy** — Live Streamlit app for real-time sentiment prediction

---

## Results

| Model | Accuracy |
|---|---|
| Logistic Regression | **85%+** |
| Naive Bayes | 80%+ |

### Key Findings
- Words like *love, amazing, excellent* are the strongest positive indicators
- Words like *terrible, broke, disappointed* strongly predict negative sentiment
- Product-level sentiment scores reveal top and underperforming products
- Verified purchases tend to have more polarized (positive or negative) reviews

---

##  How to Run

### 1. Clone the repository
```bash
git clone https://github.com/roshnitiwari1520/Amazon-Customer-Data-Analysis.git
cd Amazon-Customer-Data-Analysis
```

### 2. Install dependencies
```bash
pip install -r Requirements.txt
```

### 3. Download the dataset
- Download from [Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)
- Place `Reviews.csv` in the `data/` folder

### 4. Run the full pipeline
```bash
python run_pipeline.py
```

### 5. Launch the Streamlit app
```bash
streamlit run app.py
```

---

##  Live Predictor

Type any review and get instant sentiment + confidence score:

```
Input  : "This product is absolutely amazing! Best purchase ever."
Output : POSITIVE  |  Confidence: 94.2%
```

---

##  Future Improvements
- [ ] Add BERT / transformer-based model for higher accuracy
- [ ] Expand to multi-label emotion classification
- [ ] Add real-time review scraping from Amazon
- [ ] Deploy app on Streamlit Cloud (public URL)

---

##  Author

**Roshni Tiwari**  
🔗 [LinkedIn](https://www.linkedin.com/in/roshni-tiwari) | 🐙 [GitHub](https://github.com/roshnitiwari1520)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
