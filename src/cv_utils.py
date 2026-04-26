from sklearn.model_selection import StratifiedKFold, cross_validate
 
METRICS = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
 
def evaluate_cv(model, X, y, name: str, n_folds: int = 5) -> dict:
    #run k-fold CV and report mean ± std and also handles models that can't compute roc_auc
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    #try full metric set first, otherwise drop roc_auc if model doesn't support it
    try:
        res = cross_validate(model, X, y, cv=cv, scoring=METRICS, n_jobs=-1)
        metrics_used = METRICS
    except (AttributeError, ValueError):
        metrics_used = [m for m in METRICS if m != 'roc_auc']
        res = cross_validate(model, X, y, cv=cv, scoring=metrics_used, n_jobs=-1)

    summary = {}
    print(f"\n{name}")
    for m in metrics_used:
        mean = res[f'test_{m}'].mean()
        std = res[f'test_{m}'].std()
        summary[m] = {'mean': round(float(mean), 4), 'std': round(float(std), 4)}
        print(f"  {m:10s} {mean:.4f} (+/- {std:.4f})")

    return summary

def pick_best(results: dict, metric: str = 'f1') -> str:
    #pick the model that has the highest average score for the given metric
    return max(results, key=lambda k: results[k][metric]['mean'])