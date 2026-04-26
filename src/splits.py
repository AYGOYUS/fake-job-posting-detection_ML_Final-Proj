import os
import numpy as np
from sklearn.model_selection import train_test_split


def three_way_split(y, train_frac=0.70, val_frac=0.15, test_frac=0.15, seed=42):
    #makes a layered 70/15/15 split and return the three index arrays
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, \
        "Fractions must sum to 1"

    indices = np.arange(len(y))

    #first split will separate train data from the holdout set
    train_idx, holdout_idx = train_test_split(
        indices, test_size=(val_frac + test_frac),
        stratify=y, random_state=seed)

    #second split will divide the holdout set into validation and test sets
    val_size = val_frac / (val_frac + test_frac)
    val_idx, test_idx = train_test_split(
        holdout_idx, test_size=(1 - val_size),
        stratify=y[holdout_idx], random_state=seed)

    return train_idx, val_idx, test_idx


def report_split(y, train_idx, val_idx, test_idx, path: str = "results/split_report.txt"):
    #show the size and real/fake balance for each split
    lines = []
    lines.append(f"Total rows: {len(y):,}")
    lines.append(f"Total fake: {int(y.sum()):,}  ({100 * y.mean():.1f}%)\n")
    for name, idx in [('Train', train_idx), ('Val', val_idx), ('Test', test_idx)]:
        n = len(idx)
        n_fake = int(y[idx].sum())
        pct = 100 * n / len(y)
        fake_pct = 100 * n_fake / n if n else 0
        lines.append(f"{name:6s} {n:7,} ({pct:5.1f}%)   fake: {n_fake:5,} ({fake_pct:5.1f}%)")
    text = "\n".join(lines)
    print(text)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(text + "\n")
    print(f"\nSaved: {path}")


def save_indices(train_idx, val_idx, test_idx, folder: str = "results/splits"):
    #saves the split indices so other scripts can use them later
    os.makedirs(folder, exist_ok=True)
    np.save(f"{folder}/train_idx.npy", train_idx)
    np.save(f"{folder}/val_idx.npy", val_idx)
    np.save(f"{folder}/test_idx.npy", test_idx)
    print(f"Saved indices to: {folder}/")


def main():
    from data_loader import load_dataset
    df = load_dataset()
    y = df['fraudulent'].values

    train_idx, val_idx, test_idx = three_way_split(y)
    report_split(y, train_idx, val_idx, test_idx)
    save_indices(train_idx, val_idx, test_idx)


if __name__ == "__main__":
    main()
