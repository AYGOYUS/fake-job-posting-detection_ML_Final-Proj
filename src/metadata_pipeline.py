import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from columns import detect_columns

def build_metadata_features(df: pd.DataFrame):
    #returns feature matrix, feature names, and labels after auto-detecting column types and engineering missing-value flags
    cols = detect_columns(df)
    binary = cols['binary']
    cat = cols['categorical']
    text = cols['text']
    target = cols['target']

    print(f"Binary cols:      {binary}")
    print(f"Categorical cols: {cat}")

    df = df.copy()

    #fill missing with 0
    for col in binary:
        df[col] = df[col].fillna(0).astype(int)

    #fill missing with Unknown, then label-encode
    encoders = {}
    encoded_cat = []
    for col in cat:
        df[col] = df[col].fillna('').replace('', 'Unknown')
        enc = LabelEncoder()
        new_col = col + '_enc'
        df[new_col] = enc.fit_transform(df[col].astype(str))
        encoders[col] = enc
        encoded_cat.append(new_col)

    #make has_X features for columns that may have missing value
    has_features = []
    for col in cat + text:
        new_col = 'has_' + col
        #treat empty as missing values
        df[new_col] = (~(df[col].isna() | (df[col].astype(str).isin(['', 'Unknown'])))).astype(int)
        has_features.append(new_col)

    feature_cols = binary + encoded_cat + has_features
    X = df[feature_cols].values
    y = df[target].values

    print(f"Metadata matrix: {X.shape} ({len(feature_cols)} features)")

    #save encoders
    os.makedirs("models", exist_ok=True)
    with open("models/metadata_encoders.pkl", 'wb') as f:
        pickle.dump(encoders, f)

    return X, feature_cols, y

if __name__ == "__main__":
    from data_loader import load_dataset
    df = load_dataset()
    X, names, y = build_metadata_features(df)
    print(f"Done. Features: {X.shape}, labels: {y.shape}")
