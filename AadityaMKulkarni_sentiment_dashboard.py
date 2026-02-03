# Aaditya Kulkarni Capstone Project
# Live Twitter Sentiment Dashboard (Demo)

import pandas as pd
import re
import string
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Data (use Sentiment140 CSV)
@st.cache_data
def load_data():
    df = pd.read_csv('sentiment140.csv', encoding='latin-1', header=None) # Update with your path to the dataset
    df.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']
    df['target'] = df['target'].map({0: 0, 4: 1})  # 0=negative, 4=positive
    return df[['text', 'target']]

# 2. Preprocessing
def clean_text(text):
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'#\w+', '', text)  # Remove hashtags
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = text.encode('ascii', 'ignore').decode('ascii')  # Remove emojis
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = ' '.join([word for word in text.split() if word not in set([
        'the', 'a', 'an', 'is', 'it', 'to', 'and', 'in', 'of', 'for', 'on', 'at', 'with', 'as', 'by', 'that', 'this'
    ])])
    return text

# 3. Feature Extraction
def vectorize_text(texts):
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(texts)
    return X, vectorizer

# 4. Modeling
def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    return model

# 5. Evaluation
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return cm, f1

# 6. Streamlit Dashboard
def main():
    st.title("Live Twitter Sentiment Dashboard (Demo)")

    df = load_data()
    df['clean_text'] = df['text'].apply(clean_text)

    st.write("Sample Data", df.head())

    X, vectorizer = vectorize_text(df['clean_text'])
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    cm, f1 = evaluate_model(model, X_test, y_test)

    st.write(f"F1 Score: {f1:.2f}")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)

    st.header("Try it live!")
    user_input = st.text_input("Enter a tweet or phrase:")
    if user_input:
        cleaned = clean_text(user_input)
        X_user = vectorizer.transform([cleaned])
        pred = model.predict(X_user)[0]
        st.write("Sentiment:", "Positive" if pred == 1 else "Negative")

if __name__ == "__main__":
    main()