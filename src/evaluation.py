#visulaization/plots

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def _save_fig(fig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved: {path}")


def plot_confusion(y_true, y_pred, title: str, path: str):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'], ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual'); ax.set_title(title)
    _save_fig(fig, path)


def plot_class_distribution(y, path: str):
    counts = np.bincount(y)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(['Real', 'Fake'], counts, color=['#4CAF50', '#F44336'])
    axes[0].set_title('Class Count')
    for i, c in enumerate(counts):
        axes[0].text(i, c + max(counts)*0.01, f'{c:,}', ha='center', fontweight='bold')
    axes[1].pie(counts, labels=['Real', 'Fake'], autopct='%1.1f%%',
                colors=['#4CAF50', '#F44336'], explode=(0, 0.1))
    axes[1].set_title('Class Proportion')
    _save_fig(fig, path)


def plot_ablation(ablation: dict, path: str):
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    conditions = ['text_only', 'metadata_only', 'combined']
    labels = ['Text Only', 'Metadata Only', 'Combined']
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(conditions))
    width = 0.2

    for i, (m, color) in enumerate(zip(metrics, colors)):
        values = [ablation[c][m]['mean'] for c in conditions]
        bars = ax.bar(x + (i - 1.5) * width, values, width, label=m.capitalize(), color=color)
        for b in bars:
            ax.annotate(f'{b.get_height():.3f}',
                        (b.get_x() + b.get_width()/2, b.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', fontsize=8)

    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.15); ax.set_ylabel('Score')
    ax.set_title('Ablation Study (Text vs Metadata vs Combined)')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    _save_fig(fig, path)


def plot_model_comparison(results: dict, path: str):
    #barchart plot of f1 scores of all models
    names = list(results.keys())
    f1_means = [results[n]['f1']['mean'] for n in names]
    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(names, f1_means, color='#42A5F5')
    for b, v in zip(bars, f1_means):
        ax.text(b.get_x() + b.get_width()/2, v + 0.01, f'{v:.3f}',
                ha='center', fontsize=9)
    ax.set_ylabel('F1 Score'); ax.set_title('Model Comparison F1')
    ax.set_ylim(0, 1.1); plt.xticks(rotation=20, ha='right')
    _save_fig(fig, path)


def plot_feature_importance(importances: dict, path: str):
    items = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    names, scores = zip(*items)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(names, scores, color='#2196F3')
    ax.invert_yaxis(); ax.set_xlabel('Importance')
    ax.set_title('Top Feature Importances of Metadata Model')
    _save_fig(fig, path)


def confusion_for_all_models(text_X, meta_X, y):
    #each model is trained and its confusion matrix is plotted
    idx_train, idx_test = train_test_split(
        np.arange(len(y)), test_size=0.2, random_state=42, stratify=y)
    y_train, y_test = y[idx_train], y[idx_test]

    #Text model
    txt = LogisticRegression(max_iter=1000, class_weight='balanced',
                             random_state=42, solver='liblinear')
    txt.fit(text_X[idx_train], y_train)
    plot_confusion(y_test, txt.predict(text_X[idx_test]),
                   "Text-Only Model", "results/cm_text_model.png")

    #Metadata model
    meta = RandomForestClassifier(n_estimators=100, max_depth=15,
                                  class_weight='balanced', random_state=42)
    meta.fit(meta_X[idx_train], y_train)
    plot_confusion(y_test, meta.predict(meta_X[idx_test]),
                   "Metadata-Only Model", "results/cm_metadata_model.png")

    #Combined model
    combined_X = hstack([text_X, csr_matrix(meta_X)])
    comb = LogisticRegression(max_iter=1000, class_weight='balanced',
                              random_state=42, solver='liblinear')
    comb.fit(combined_X.tocsr()[idx_train], y_train)
    plot_confusion(y_test, comb.predict(combined_X.tocsr()[idx_test]),
                   "Combined Model", "results/cm_combined_model.png")


def run_evaluation(text_X, meta_X, y):
    print("\n=== Evaluation ===")
    plot_class_distribution(y, "results/class_distribution.png")
    confusion_for_all_models(text_X, meta_X, y)

    #total comparisons between all models trained
    all_results = {}
    for fname in ['text_model_results.json', 'metadata_model_results.json',
                  'combined_model_results.json']:
        path = f"results/{fname}"
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        for k, v in data.items():
            if isinstance(v, dict) and 'f1' in v:
                all_results[k] = v
    if all_results:
        plot_model_comparison(all_results, "results/model_comparison.png")

    #ablation
    cpath = "results/combined_model_results.json"
    if os.path.exists(cpath):
        with open(cpath) as f:
            data = json.load(f)
        if 'ablation_study' in data:
            plot_ablation(data['ablation_study'], "results/ablation_study.png")

    #feature importance
    mpath = "results/metadata_model_results.json"
    if os.path.exists(mpath):
        with open(mpath) as f:
            data = json.load(f)
        if 'feature_importance' in data:
            plot_feature_importance(data['feature_importance'],
                                    "results/feature_importance.png")


if __name__ == "__main__":
    from data_loader import load_dataset
    from text_pipeline import build_text_features
    from metadata_pipeline import build_metadata_features
    df = load_dataset()
    text_X, _, y = build_text_features(df)
    meta_X, _, _ = build_metadata_features(df)
    run_evaluation(text_X, meta_X, y)
