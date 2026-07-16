"""
SHAP explainability for the winning model from /train.

Uses TreeExplainer(feature_perturbation="tree_path_dependent") for tree
ensembles (Random Forest, XGBoost) and the generic shap.Explainer for
linear models (Linear/Logistic Regression). Tree-path-dependent mode is
used deliberately rather than the "interventional" default: newer
XGBoost versions (3.x) mark integer/bool feature columns as categorical
internally, which the interventional path doesn't support and raises
`NotImplementedError: Categorical split is not yet supported`.
tree_path_dependent avoids this entirely and needs no background dataset.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier

MAX_SAMPLE_ROWS = 100  # SHAP on tree ensembles can be slow; a sample is enough for ranking

TREE_MODEL_TYPES = (RandomForestRegressor, RandomForestClassifier, XGBRegressor, XGBClassifier)


def compute_feature_importance(model, X: pd.DataFrame) -> tuple[list[tuple[str, float]], int]:
    """
    Returns (sorted list of (feature_name, mean_abs_shap_value), n_rows_sampled).
    Handles regression (2D shap values), binary and multiclass classification
    (3D shap values: samples x features x classes) uniformly.
    """
    if len(X) > MAX_SAMPLE_ROWS:
        X_sample = X.sample(n=MAX_SAMPLE_ROWS, random_state=42)
    else:
        X_sample = X

    if isinstance(model, TREE_MODEL_TYPES):
        explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
        shap_values = explainer(X_sample, check_additivity=False)
    else:
        explainer = shap.Explainer(model, X_sample)
        shap_values = explainer(X_sample)

    values = shap_values.values  # ndarray

    if values.ndim == 3:
        # (n_samples, n_features, n_classes) -> average abs over samples AND classes
        importance = np.abs(values).mean(axis=(0, 2))
    else:
        # (n_samples, n_features) -> average abs over samples
        importance = np.abs(values).mean(axis=0)

    pairs = list(zip(X.columns.tolist(), importance.tolist()))
    pairs.sort(key=lambda p: p[1], reverse=True)

    return pairs, len(X_sample)