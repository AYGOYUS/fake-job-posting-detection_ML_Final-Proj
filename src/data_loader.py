import os
import sys
import pandas as pd

DEFAULT_PATH = "data/fake_job_postings.csv"


def load_dataset(path: str = DEFAULT_PATH) -> pd.DataFrame:
    #checks the data file, will exit if not found.
    if not os.path.exists(path):
        print(f"ERROR: '{path}' not found. Download the csv file from kaggle into data folder.")
        sys.exit(1)
    return pd.read_csv(path)


def summarize(df: pd.DataFrame) -> None:
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns\n") #printing datasets characteristics

    #missing values per column (NaN + empty strings)
    print("Missing values:")
    for col in df.columns:
        miss = df[col].isna().sum()
        if df[col].dtype == 'object':
            miss += (df[col].fillna('') == '').sum() - df[col].isna().sum()
        if miss > 0:
            pct = 100 * miss / len(df)
            print(f"  {col:25s} {miss:6d} ({pct:5.1f}%)")

    #class balance
    if 'fraudulent' in df.columns:
        counts = df['fraudulent'].value_counts().sort_index()
        print(f"\nClass distribution:")
        for label, n in counts.items():
            name = "Real" if label == 0 else "Fake"
            print(f"  {name} ({label}): {n:,} ({100*n/len(df):.1f}%)")


if __name__ == "__main__":
    df = load_dataset()
    summarize(df)
