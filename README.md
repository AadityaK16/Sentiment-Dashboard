# Twitter Sentiment Dashboard

An interactive Streamlit app for binary sentiment classification of tweets, trained on the Sentiment140 dataset (1.6M tweets). Features a full NLP preprocessing pipeline, TF-IDF vectorization, logistic regression classifier, live model evaluation, and a real-time inference widget for user-entered text.

---

## Overview

Sentiment analysis maps raw, noisy social media text to a binary label — positive or negative. This project handles the full pipeline: cleaning Twitter-specific noise (mentions, hashtags, URLs, emojis), extracting TF-IDF features, training a logistic regression classifier, and deploying it in a Streamlit dashboard where users can type any phrase and get an instant prediction.

---

## Features

- **Twitter-aware text preprocessing** — strips mentions, hashtags, URLs, emojis, punctuation, digits, and common stopwords
- **TF-IDF vectorization** — top 5,000 features from cleaned text
- **Logistic Regression classifier** — fast, interpretable baseline well-suited to sparse TF-IDF input
- **Evaluation** — F1 score + confusion matrix heatmap (Seaborn) rendered in dashboard
- **Live inference widget** — enter any text and receive a Positive/Negative prediction in real time

---

## Model Details

| Component | Choice | Rationale |
|---|---|---|
| Vectorizer | TF-IDF, `max_features=5000` | Sparse, fast, strong baseline for short text |
| Classifier | Logistic Regression, `max_iter=200` | Linear models work well on high-dimensional sparse features; interpretable |
| Train/test split | 80/20, `random_state=42` | Standard holdout evaluation |
| Evaluation metric | F1 score | Better than accuracy for potentially skewed label distributions |

### Preprocessing steps (in order)
1. Remove `@mentions`
2. Remove `#hashtags`
3. Remove URLs (`http...`)
4. Strip non-ASCII characters (emojis)
5. Lowercase
6. Remove punctuation
7. Remove digits
8. Remove common stopwords (lightweight custom list)

---

## Dashboard

Built with Streamlit:

1. **Sample data view** — displays first rows of loaded dataset
2. **F1 Score** — printed after model evaluation on test set
3. **Confusion matrix** — Seaborn heatmap (true label × predicted label)
4. **Live prediction widget** — text input → cleaned → vectorized → predicted → displayed as "Positive" or "Negative"

---

## Tech Stack

- **NLP & ML:** Scikit-learn (TF-IDF, Logistic Regression, evaluation metrics)
- **Data:** Pandas, re, string
- **Dashboard:** Streamlit
- **Visualization:** Matplotlib, Seaborn

---

## Setup & Usage

### 1. Install dependencies
```bash
pip install pandas scikit-learn streamlit matplotlib seaborn
```

### 2. Dataset
Uses [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140) — 1.6M labeled tweets (0 = negative, 4 = positive, remapped to 0/1).

Download from Kaggle and update the path:
```python
df = pd.read_csv('sentiment140.csv', encoding='latin-1', header=None)
```

### 3. Run the dashboard
```bash
streamlit run AadityaMKulkarni_sentiment_dashboard.py
```

App opens at `http://localhost:8501`.

---

## Key Design Decisions

**Why Logistic Regression over a neural network?** LR is a strong, fast baseline on TF-IDF features — high-dimensional sparse input is exactly where linear models excel. For a 1.6M-row dataset without GPU resources, LR trains in seconds and achieves competitive F1 scores. A transformer (e.g. DistilBERT) would improve contextual understanding but would require significantly more compute.

**Why TF-IDF over bag-of-words?** TF-IDF down-weights frequent terms across documents (like "the", "and") automatically, without relying entirely on a hardcoded stopword list. Combined with the custom stopword removal step, this produces cleaner feature representations.

**Why `@st.cache_data`?** Loading and cleaning 1.6M rows is expensive. Streamlit re-runs the full script on every interaction — `@st.cache_data` ensures the data loads once and is reused across UI interactions.

---

## Limitations & Next Steps

- The custom stopword list is minimal; a full NLTK stopword list would improve feature quality
- TF-IDF loses word order and context — a fine-tuned transformer (DistilBERT, RoBERTa) would handle sarcasm and negation better
- The model is trained once at startup; a production version would serialize the vectorizer and model with `joblib` to avoid retraining on every session

---

## Author
Aaditya Kulkarni — [GitHub](https://github.com/AadityaK16) · [LinkedIn](https://www.linkedin.com/in/aaditya-kulkarni-06932b32a/)
