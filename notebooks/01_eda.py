import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from data_loader import load_dataset
from columns import detect_columns

os.makedirs("results", exist_ok=True)


def overview(df):
    print("\n Data Overview ")
    print(f"Rows: {df.shape[0]:,}    Columns: {df.shape[1]}")
    cols = detect_columns(df)
    for kind in ['text', 'binary', 'categorical']:
        print(f"{kind:12s} ({len(cols[kind])}): {cols[kind]}")


def missing_values(df):  #counts missing values in each column and save a chart for columns that have them
    print("\n Missing Values ")
    rows = []
    for col in df.columns:
        miss = df[col].isna().sum()
        if df[col].dtype == 'object':
            miss = (df[col].isna() | (df[col] == '')).sum()
        if miss > 0:
            rows.append((col, miss, 100 * miss / len(df)))
    rows.sort(key=lambda r: r[2], reverse=True)
    for col, miss, pct in rows:
        print(f"  {col:25s} {miss:6d} ({pct:5.1f}%)")

    if rows:
        fig, ax = plt.subplots(figsize=(10, max(3, 0.4 * len(rows))))
        names, _, pcts = zip(*rows)
        ax.barh(names, pcts, color='#FF7043')
        ax.invert_yaxis(); ax.set_xlabel('% Missing')
        ax.set_title('Missing Values by Column')
        plt.tight_layout()
        plt.savefig("results/eda_missing_values.png", dpi=150, bbox_inches='tight')
        plt.close()


def class_distribution(df): #how many real and fake posts there are? then save count and percentage charts
    print("\n Class Distribution ")
    counts = df['fraudulent'].value_counts().sort_index()
    total = len(df)
    for label, n in counts.items():
        name = "Real" if label == 0 else "Fake"
        print(f"  {name}: {n:,} ({100 * n / total:.1f}%)")
    if 0 in counts and 1 in counts:
        print(f" Imbalance ratio: {counts[0] / counts[1]:.1f} : 1")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(['Real', 'Fake'], counts.values, color=['#4CAF50', '#F44336'])
    axes[0].set_title('Class Count')
    axes[1].pie(counts.values, labels=['Real', 'Fake'], autopct='%1.1f%%',
                colors=['#4CAF50', '#F44336'], explode=(0, 0.1))
    axes[1].set_title('Class Proportion')
    plt.tight_layout()
    plt.savefig("results/eda_class_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()


def text_length_comparison(df):
   #comparing the average text length of each column for real vs fake post
    print("\n Text Length: Real vs Fake ")
    cols = detect_columns(df)
    text_cols = cols['text']
    if not text_cols:
        return

    real = df[df['fraudulent'] == 0]
    fake = df[df['fraudulent'] == 1]

    print(f"  {'column':25s} {'real':>8s} {'fake':>8s} {'diff':>8s}")
    real_means, fake_means = [], []
    for col in text_cols:
        r = real[col].fillna('').str.len().mean()
        f = fake[col].fillna('').str.len().mean()
        real_means.append(r); fake_means.append(f)
        print(f"  {col:25s} {r:8.1f} {f:8.1f} {f - r:+8.1f}")

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(text_cols)); w = 0.35
    ax.bar(x - w/2, real_means, w, label='Real', color='#4CAF50')
    ax.bar(x + w/2, fake_means, w, label='Fake', color='#F44336')
    ax.set_xticks(x); ax.set_xticklabels(text_cols, rotation=15, ha='right')
    ax.set_ylabel('Avg Length (chars)'); ax.set_title('Text Length: Real vs Fake')
    ax.legend(); plt.tight_layout()
    plt.savefig("results/eda_text_length.png", dpi=150, bbox_inches='tight')
    plt.close()


def feature_differences(df):
    #comparing binary columns and missing rates for real vs fake post
    print("\n Feature Differences ")
    cols = detect_columns(df)
    binary = cols['binary']

    real = df[df['fraudulent'] == 0]
    fake = df[df['fraudulent'] == 1]

    if binary:
        print("\n  Binary feature means:")
        print(f"  {'feature':25s} {'real':>8s} {'fake':>8s} {'diff':>8s}")
        for col in binary:
            r, f = real[col].mean(), fake[col].mean()
            print(f"  {col:25s} {r:8.3f} {f:8.3f} {f - r:+8.3f}")

    print("\n  Missing rate differences (>2%):")
    print(f"  {'feature':25s} {'real':>8s} {'fake':>8s} {'diff':>8s}")
    diffs = []
    for col in df.columns:
        if col in ('job_id', 'fraudulent'):
            continue
        r_miss = (real[col].isna() | (real[col].astype(str) == '')).mean()
        f_miss = (fake[col].isna() | (fake[col].astype(str) == '')).mean()
        d = f_miss - r_miss
        if abs(d) > 0.02:
            diffs.append((col, r_miss, f_miss, d))
            print(f"  {col:25s} {r_miss:8.3f} {f_miss:8.3f} {d:+8.3f}")


def correlations(df):
    #checks how numeric and derived features relate to the target
    print("\n Correlations with target ")
    df = df.copy()

    #Add has_X indicators for every non-numeric column
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[f'has_{col}'] = (~(df[col].isna() | (df[col].astype(str) == ''))).astype(int)
    if 'description' in df.columns:
        df['desc_length'] = df['description'].fillna('').astype(str).str.len()

    numeric = df.select_dtypes(include=['int64', 'float64'])
    if 'job_id' in numeric.columns:
        numeric = numeric.drop(columns=['job_id'])

    #drops columns with the same value everywhere, since their correlation would be NaN.
    numeric = numeric.loc[:, numeric.nunique() > 1]

    corrs = numeric.corr()['fraudulent'].drop('fraudulent').dropna().sort_values()
    for feat, c in corrs.items():
        bar = '#' * int(abs(c) * 30)
        print(f"  {feat:30s} {c:+.4f}  {bar}")

    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(corrs))))
    colors = ['#F44336' if v < 0 else '#4CAF50' for v in corrs.values]
    ax.barh(corrs.index, corrs.values, color=colors)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Correlation with fraudulent')
    ax.set_title('Feature Correlation with Target')
    plt.tight_layout()
    plt.savefig("results/eda_correlation.png", dpi=150, bbox_inches='tight')
    plt.close()


def auto_summary(df):
    #All results here are calculated from the data, not hardcoded
    print("\n Auto Summary ")
    real = df[df['fraudulent'] == 0]
    fake = df[df['fraudulent'] == 1]
    fake_pct = 100 * len(fake) / len(df)
    ratio = len(real) / max(len(fake), 1)
    print(f"  Fake postings: {len(fake):,} ({fake_pct:.1f}%), ratio {ratio:.1f}:1")

    cols = detect_columns(df)

    #Binary features with the biggest real vs. fake differences
    if cols['binary']:
        diffs = [(c, fake[c].mean() - real[c].mean()) for c in cols['binary']]
        diffs.sort(key=lambda x: abs(x[1]), reverse=True)
        print("  Top discriminative binary features:")
        for c, d in diffs[:3]:
            direction = "higher in fake" if d > 0 else "lower in fake"
            print(f"    {c}: {abs(d):.3f} ({direction})")

    #Text columns with the biggest length differences
    if cols['text']:
        tdiffs = []
        for c in cols['text']:
            r = real[c].fillna('').str.len().mean()
            f = fake[c].fillna('').str.len().mean()
            if r > 0:
                tdiffs.append((c, (f - r) / r * 100))
        tdiffs.sort(key=lambda x: abs(x[1]), reverse=True)
        if tdiffs:
            print(" Largest text length differences:")
            for c, d in tdiffs[:3]:
                print(f"    {c}: fake is {abs(d):.0f}% {'longer' if d > 0 else 'shorter'} than real")


if __name__ == "__main__":
    df = load_dataset()
    overview(df)
    missing_values(df)
    class_distribution(df)
    text_length_comparison(df)
    feature_differences(df)
    correlations(df)
    auto_summary(df)
    print("\nPlots saved to results/")
