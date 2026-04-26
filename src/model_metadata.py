import os
import json
import pickle
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from cv_utils import evaluate_cv, pick_best

def get_models():
    #return a set of tree-based models with fixed hyperparameters
    return {
        'Decision Tree': DecisionTreeClassifier(
            max_depth=10, class_weight='balanced', random_state=42),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, max_depth=15,
            class_weight='balanced', random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100, max_depth=5,
            learning_rate=0.1, random_state=42),
    }

def top_features(model, feature_names: list, k: int = 10) -> dict:
    #get top k most important features from a tree-based model
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1][:k]
    return {feature_names[i]: round(float(importances[i]), 4) for i in idx}

def train_metadata_models(X, y, feature_names) -> tuple[dict, str]:
    #run cross-validation for each model, save best one, and show feature importance
    models = get_models()
    results = {name: evaluate_cv(m, X, y, name) for name, m in models.items()}

    best_name = pick_best(results)
    print(f"\nBest metadata model: {best_name} (F1={results[best_name]['f1']['mean']:.4f})")

    best = models[best_name].fit(X, y)
    importances = top_features(best, feature_names)
    print(f"\nTop features:")
    for name, score in importances.items():
        print(f"  {name:30s} {score:.4f}")

    results['feature_importance'] = importances

    os.makedirs("models", exist_ok=True)
    with open("models/best_metadata_model.pkl", 'wb') as f:
        pickle.dump(best, f)

    os.makedirs("results", exist_ok=True)
    with open("results/metadata_model_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    return results, best_name

if __name__ == "__main__":
    from data_loader import load_dataset
    from metadata_pipeline import build_metadata_features
    df = load_dataset()
    X, names, y = build_metadata_features(df)
    train_metadata_models(X, y, names)
