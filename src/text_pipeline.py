#Combine text columns, clean text, build TF-IDF features.

import os
import re
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.feature_selection import SelectKBest, chi2

from columns import detect_columns

STOP_WORDS = set(ENGLISH_STOP_WORDS)


def clean_text(text: str) -> str:
    #cleaning the text by making it lowercase, removing links/symbols, and dropping stopwords or tiny words.
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    words = [w for w in text.split() if w not in STOP_WORDS and len(w) > 2]
    return ' '.join(words)

def build_text_features(df: pd.DataFrame, max_features: int = 5000, k_best: int = 2000):
    #Build TF-IDF features, select the best ones with chi-squared, and return the features, vectorizer, and labels
    #The text columns are detected automatically
    cols = detect_columns(df)
    text_cols = cols['text']
    target = cols['target']

    print(f"Text columns detected: {text_cols}")

    #combine all text columns into one string for each row
    merged = df[text_cols].fillna('').agg(' '.join, axis=1)
    cleaned = merged.apply(clean_text)

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
    )
    X_full = vectorizer.fit_transform(cleaned)
    y = df[target].values

    #top-k features by chi-squared score against the target and stops k at the number of features we actually have
    k = min(k_best, X_full.shape[1])
    selector = SelectKBest(chi2, k=k)
    X = selector.fit_transform(X_full, y)
    print(f"TF-IDF: {X_full.shape}  ->  After SelectKBest(k={k}): {X.shape}")

    #saves the vectorizer and selector to reuse them later
    os.makedirs("models", exist_ok=True)
    with open("models/tfidf_vectorizer.pkl", 'wb') as f:
        pickle.dump(vectorizer, f)
    with open("models/feature_selector.pkl", 'wb') as f:
        pickle.dump(selector, f)

    return X, vectorizer, y


if __name__ == "__main__":
    from data_loader import load_dataset
    df = load_dataset()
    X, vec, y = build_text_features(df)
    print(f"Done. Features: {X.shape}, labels: {y.shape}")
