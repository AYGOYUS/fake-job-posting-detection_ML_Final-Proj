#Analyzes combined-model job-posting misclassifications by counting false positives/negatives, showing representative examples, and summarizing common error-group words.

import os
import json
import numpy as np
import pandas as pd
from collections import Counter
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from columns import detect_columns


def merge_text(row, text_cols) -> str:
    parts = [str(row[c]) for c in text_cols if pd.notna(row[c])] #all text columns of a single row is merged into one string
    return ' '.join(parts)


def common_words(texts, top_n: int = 10) -> dict:
    #the most common content words in a list of strings is returned
    words = []
    for t in texts:
        for w in str(t).lower().split():
            w = ''.join(c for c in w if c.isalpha())
            if len(w) > 3 and w not in ENGLISH_STOP_WORDS:
                words.append(w)
    return dict(Counter(words).most_common(top_n))


def show_examples(group: pd.DataFrame, text_cols: list, label: str, n: int = 3) -> None:
    #shows a example postings from missclassified group
    print(f"\n--- {label} ({len(group)} total) ---")
    for _, row in group.head(n).iterrows():
        print(f"\n  title: {str(row.get('title', ''))[:80]}")
        text = merge_text(row, text_cols)
        print(f"  text:  {text[:200]}...")


def main():
    from data_loader import load_dataset
    from text_pipeline import build_text_features
    from metadata_pipeline import build_metadata_features

    df = load_dataset()
    cols = detect_columns(df)

    #feature builidng
    text_X, _, y = build_text_features(df)
    meta_X, _, _ = build_metadata_features(df)
    X = hstack([text_X, csr_matrix(meta_X)]).tocsr()

    #Train/test split
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        idx, test_size=0.2, random_state=42, stratify=y)

    #trainig combined models
    model = LogisticRegression(
        max_iter=1000, class_weight='balanced',
        random_state=42, solver='liblinear')
    model.fit(X[train_idx], y[train_idx])
    y_pred = model.predict(X[test_idx]) #prediction
    y_true = y[test_idx]

    #erros identification
    df_test = df.iloc[test_idx].reset_index(drop=True)
    df_test['actual'] = y_true
    df_test['predicted'] = y_pred
    fp = df_test[(df_test['actual'] == 0) & (df_test['predicted'] == 1)]
    fn = df_test[(df_test['actual'] == 1) & (df_test['predicted'] == 0)]

    print(f"\nTest set:        {len(y_true):,}")
    print(f"False positives: {len(fp)}  (real postings flagged as fake)")
    print(f"False negatives: {len(fn)}  (fake postings missed)")

    show_examples(fp, cols['text'], "FALSE POSITIVES")
    show_examples(fn, cols['text'], "FALSE NEGATIVES")

    #common words found in each error group
    fp_words = common_words(fp.apply(lambda r: merge_text(r, cols['text']), axis=1)) if len(fp) else {}
    fn_words = common_words(fn.apply(lambda r: merge_text(r, cols['text']), axis=1)) if len(fn) else {}

    print(f"\nTop words in FALSE POSITIVES: {list(fp_words.keys())[:10]}")
    print(f"Top words in FALSE NEGATIVES: {list(fn_words.keys())[:10]}")

    #saved to results
    summary = {
        'test_size': int(len(y_true)),
        'false_positives': int(len(fp)),
        'false_negatives': int(len(fn)),
        'fp_common_words': fp_words,
        'fn_common_words': fn_words,
    }
    os.makedirs('results', exist_ok=True)
    with open('results/error_analysis.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("\nSaved: results/error_analysis.json")


if __name__ == "__main__":
    main()
