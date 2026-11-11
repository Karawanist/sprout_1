"""Prediction helpers. Tree spread is disagreement, not a confidence interval."""
import numpy as np

def predict_with_range(model, X):
    values = X.values if hasattr(X, "values") else np.asarray(X)
    preds = np.asarray([tree.predict(values) for tree in model.estimators_])
    return preds.mean(axis=0), np.percentile(preds, 10, axis=0), np.percentile(preds, 90, axis=0)
