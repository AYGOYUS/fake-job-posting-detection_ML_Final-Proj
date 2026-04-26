#here we train and evaluate the text only models

import os
import json
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from cv_utils import evaluate_cv, pick_best


def get_models():
    #Uses three standard text classifiers, with class weights balanced to handle uneven classes
    return {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, class_weight='balanced',
            random_state=42, solver='liblinear'),
        'Naive Bayes': MultinomialNB(alpha=1.0),
        'Linear SVM': LinearSVC(
            max_iter=2000, class_weight='balanced',
            random_state=42, C=1.0),
    }


def train_text_models(X, y) -> tuple[dict, str]:
    #Run cross validation for each model, save the best performer, and return its results and name
    models = get_models()
    results = {name: evaluate_cv(m, X, y, name) for name, m in models.items()}

    best_name = pick_best(results)
    print(f"\nBest text model: {best_name} (F1={results[best_name]['f1']['mean']:.4f})")

    #Trains on whole data and saves the results
    best = models[best_name].fit(X, y)
    os.makedirs("models", exist_ok=True)
    with open("models/best_text_model.pkl", 'wb') as f:
        pickle.dump(best, f)

    os.makedirs("results", exist_ok=True)
    with open("results/text_model_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    return results, best_name


if __name__ == "__main__":
    from data_loader import load_dataset
    from text_pipeline import build_text_features
    df = load_dataset()
    X, _, y = build_text_features(df)
    train_text_models(X, y)
