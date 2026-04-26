from sklearn.model_selection import StratifiedKFold, cross_validate

METRICS = ['accuracy', 'precision', 'recall', 'f1']


def evaluate_cv(model, X, y, name: str, n_folds: int = 5) -> dict:
    #run stratified k-fold cross validation and report mean and std for each metric
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    res = cross_validate(model, X, y, cv=cv, scoring=METRICS, n_jobs=-1)

    summary = {}
    print(f"\n{name}")
    for m in METRICS:
        mean = res[f'test_{m}'].mean()
        std = res[f'test_{m}'].std()
        summary[m] = {'mean': round(float(mean), 4), 'std': round(float(std), 4)}
        print(f"  {m:10s} {mean:.4f} (+/- {std:.4f})")
    return summary

def pick_best(results: dict, metric: str = 'f1') -> str:
    #pick the model with the highest mean score for the given metric
    return max(results, key=lambda k: results[k][metric]['mean'])