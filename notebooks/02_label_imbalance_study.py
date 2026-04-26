import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from data_loader import load_dataset

os.makedirs("results", exist_ok=True)


def class_proportions(df):
    counts = df['fraudulent'].value_counts().sort_index()
    total = len(df)
    print(f"\nReal: {counts[0]:,} ({100*counts[0]/total:.1f}%)")
    print(f"Fake: {counts[1]:,} ({100*counts[1]/total:.1f}%)")
    print(f"Ratio: {counts[0]/counts[1]:.1f} : 1")
    return counts


def plot_distribution(counts):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    colors = ["#0EF116", "#470803"]
    axes[0].bar(['Real', 'Fake'], counts, color=colors)
    axes[0].set_title('Count'); axes[0].set_ylabel('Postings')
    axes[1].pie(counts, labels=['Real', 'Fake'], autopct='%1.1f%%',
                colors=colors, explode=(0, 0.1))
    axes[1].set_title('Proportion')
    axes[2].bar(['Real', 'Fake'], counts, color=colors); axes[2].set_yscale('log')
    axes[2].set_title('Log Scale'); axes[2].set_ylabel('Postings (log)')
    plt.tight_layout()
    plt.savefig("results/imbalance_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()

def accuracy_trap(counts):
    #shows why a model that predicts only real can still look accurate but is actually useless
    total = counts.sum()
    print("\n--- The Accuracy Trap ---")
    print("If a model predicts 'Real' for every posting:")
    print(f"  Accuracy:  {counts[0]/total:.4f}  (looks great)")
    print(f"  Recall:    0.0000  (catches zero fakes)")
    print(f"  F1 score:  0.0000  (reveals it's useless)")
    print("\n=> We use F1 score as the primary metric, not accuracy.")

def metric_explanations():
    print("\n--- Metric Choices ---")
    print("Precision: of postings marked fake, how many really are?")
    print("Recall:    of all real fakes, how many did we actually catch?")
    print("F1:        balances precision and recall (primary metric)")
    print("\nHandling imbalance:")
    print("  - class_weight='balanced' in sklearn models")
    print("  - SMOTE oversampling on training folds")
    print("  - Stratified k-fold so every fold has both classes")


if __name__ == "__main__":
    df = load_dataset()
    counts = class_proportions(df)
    plot_distribution(counts)
    accuracy_trap(counts)
    metric_explanations()
    print("\nDone. See results/imbalance_analysis.png")
