import os
import json
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def tune(model, grid, X, y, name: str) -> dict:
    #run grid search using F1 score and return the best params and score
    print(f"\n--- Tuning {name} ---")
    gs = GridSearchCV(model, grid, scoring='f1', cv=CV, n_jobs=-1, verbose=1)
    gs.fit(X, y)
    print(f"Best params: {gs.best_params_}")
    print(f"Best F1:     {gs.best_score_:.4f}")
    return {
        'best_params': gs.best_params_,
        'best_f1': round(float(gs.best_score_), 4),
    }

def tune_text(X, y) -> dict:
    #search for best C and class weight for logistic regression on text data
    base = LogisticRegression(max_iter=1000, random_state=42, solver='liblinear')
    grid = {
        'C': [0.5, 1.0, 2.0],
        'class_weight': ['balanced', None],
    }
    return tune(base, grid, X, y, "Logistic Regression (text)")


def tune_metadata(X, y) -> dict:
    #search for best number of trees and depth for random forest on metadata
    base = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)
    grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 15, 20],
    }
    return tune(base, grid, X, y, "Random Forest (metadata)")


def main():
    from data_loader import load_dataset
    from text_pipeline import build_text_features
    from metadata_pipeline import build_metadata_features

    df = load_dataset()
    text_X, _, y = build_text_features(df)
    meta_X, _, _ = build_metadata_features(df)

    results = {
        'text': tune_text(text_X, y),
        'metadata': tune_metadata(meta_X, y),
    }

    os.makedirs('results', exist_ok=True)
    with open('results/tuning_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n--- Summary ---")
    print(f"Text     best F1: {results['text']['best_f1']}  params: {results['text']['best_params']}")
    print(f"Metadata best F1: {results['metadata']['best_f1']}  params: {results['metadata']['best_params']}")
    print("\nSaved: results/tuning_results.json")


if __name__ == "__main__":
    main()
