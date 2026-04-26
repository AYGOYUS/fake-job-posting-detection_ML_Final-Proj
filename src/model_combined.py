import os
import json
import pickle
from scipy.sparse import hstack, csr_matrix
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from cv_utils import evaluate_cv, pick_best

def fuse(text_X, meta_X):
    #combine text features and metadata features into one matrix
    return hstack([text_X, csr_matrix(meta_X)])

def get_combined_models():
    #define a set of models to use on combined feature set
    return {
        'Combined LR': LogisticRegression(
            max_iter=1000, class_weight='balanced',
            random_state=42, solver='liblinear'),
        'Combined RF': RandomForestClassifier(
            n_estimators=100, max_depth=20,
            class_weight='balanced', random_state=42, n_jobs=-1),
        'Combined SVM': LinearSVC(
            max_iter=2000, class_weight='balanced',
            random_state=42, C=1.0),
    }

def ablation_study(text_X, meta_X, y) -> dict:
    #compare performance using text only, metadata only, and combined features
    base = LogisticRegression(
        max_iter=1000, class_weight='balanced',
        random_state=42, solver='liblinear')

    print("\n=== Ablation Study (Logistic Regression) ===")
    return {
        'text_only': evaluate_cv(clone(base), text_X, y, "Text Only"),
        'metadata_only': evaluate_cv(clone(base), meta_X, y, "Metadata Only"),
        'combined': evaluate_cv(clone(base), fuse(text_X, meta_X), y, "Combined"),
    }

def train_combined_models(text_X, meta_X, y) -> tuple[dict, str]:
    #train models on combined features, run ablation study, and save outputs
    combined_X = fuse(text_X, meta_X)
    print(f"Combined feature matrix: {combined_X.shape}")

    models = get_combined_models()
    results = {name: evaluate_cv(m, combined_X, y, name) for name, m in models.items()}

    best_name = pick_best(results)
    print(f"\nBest combined model: {best_name} (F1={results[best_name]['f1']['mean']:.4f})")

    best = models[best_name].fit(combined_X, y)
    os.makedirs("models", exist_ok=True)
    with open("models/best_combined_model.pkl", 'wb') as f:
        pickle.dump(best, f)

    #run ablation study to compare feature sets
    results['ablation_study'] = ablation_study(text_X, meta_X, y)

    os.makedirs("results", exist_ok=True)
    with open("results/combined_model_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    return results, best_name

if __name__ == "__main__":
    from data_loader import load_dataset
    from text_pipeline import build_text_features
    from metadata_pipeline import build_metadata_features
    df = load_dataset()
    text_X, _, y = build_text_features(df)
    meta_X, _, _ = build_metadata_features(df)
    train_combined_models(text_X, meta_X, y)
