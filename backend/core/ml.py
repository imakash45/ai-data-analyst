"""
/train pipeline logic.

Design (per the counter-questions this followed):
- All 3 models (Linear/Logistic Regression, Random Forest, XGBoost) are
  trained on the EXACT SAME encoded feature matrix, so the comparison
  table is apples-to-apples — no per-model encoding differences.
- Encoding rule is deterministic per column, based on inferred_type +
  cardinality (not a user choice, unlike /clean's imputation strategy):
    numeric      -> used as-is (safety median-fill if still NaN)
    boolean      -> mapped to 0/1
    categorical  -> one-hot encoded IF nunique <= CATEGORICAL_ONEHOT_LIMIT,
                    else dropped (too many dummy columns would be noise)
    text         -> always dropped (free text, not tabular ML input)
    datetime     -> always dropped (date feature engineering is out of
                    scope for this module)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    r2_score,
    root_mean_squared_error,
    accuracy_score,
    f1_score,
    precision_score,
)
from xgboost import XGBRegressor, XGBClassifier

from core.schemas import ColumnInfo, DroppedColumn, ModelMetrics, ModelResult, TaskType

CATEGORICAL_ONEHOT_LIMIT = 20
TEST_SIZE = 0.2
RANDOM_STATE = 42

CATEGORICAL_ONEHOT_LIMIT = 20
LABEL_ENCODE_MAX_UNIQUE = 200  # above this, treat as ID-like and drop entirely

class TrainingError(Exception):
    """Raised for user-facing validation problems (bad target column, etc.)."""
    pass


def detect_task_type(target_info: ColumnInfo) -> TaskType:
    if target_info.inferred_type in ("categorical", "boolean"):
        return "classification"
    if target_info.inferred_type == "numeric":
        return "regression"
    # text / datetime targets aren't supported
    raise TrainingError(
        f"Column '{target_info.name}' is inferred as '{target_info.inferred_type}', "
        "which isn't usable as an ML target. Pick a numeric column for regression "
        "or a categorical/boolean column for classification."
    )


def prepare_features(
    df: pd.DataFrame,
    target_column: str,
    columns: list[ColumnInfo],
    excluded_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[DroppedColumn], TaskType, list | None]:
    excluded_columns = set(excluded_columns or [])
    column_by_name = {c.name: c for c in columns}
    if target_column not in column_by_name:
        raise TrainingError(f"Column '{target_column}' not found in dataset.")

    target_info = column_by_name[target_column]
    task_type = detect_task_type(target_info)

    work_df = df.dropna(subset=[target_column]).copy()

    dropped: list[DroppedColumn] = []
    feature_frames: list[pd.DataFrame] = []

    for col_info in columns:
        col = col_info.name
        if col == target_column:
            continue

        if col in excluded_columns:
            dropped.append(DroppedColumn(column=col, reason="manually excluded by user before training"))
            continue

        if col_info.inferred_type == "text":
            dropped.append(DroppedColumn(column=col, reason="free text column, not usable as a model feature"))
            continue

        if col_info.inferred_type == "datetime":
            dropped.append(DroppedColumn(column=col, reason="datetime feature engineering not in scope for this module"))
            continue

        if col_info.inferred_type == "boolean":
            feature_frames.append(work_df[[col]].astype(int))
            continue

        if col_info.inferred_type == "numeric":
            series = pd.to_numeric(work_df[col], errors="coerce")
            if series.isna().any():
                series = series.fillna(series.median())
            feature_frames.append(series.to_frame(col))
            continue

        if col_info.inferred_type == "categorical":
            n_unique = work_df[col].nunique(dropna=True)

            if n_unique > LABEL_ENCODE_MAX_UNIQUE:
                dropped.append(
                    DroppedColumn(
                        column=col,
                        reason=f"too many categories ({n_unique} > {LABEL_ENCODE_MAX_UNIQUE}) even for label encoding — likely an ID-like column",
                    )
                )
                continue

            if n_unique > CATEGORICAL_ONEHOT_LIMIT:
                # too many categories for one-hot (would blow up feature count),
                # but not so many that it looks like a unique ID column ->
                # label encode instead of dropping, so the column still contributes signal
                codes, _ = pd.factorize(work_df[col].astype(str))
                feature_frames.append(pd.Series(codes, index=work_df.index, name=col).to_frame())
                continue

            dummies = pd.get_dummies(work_df[col].astype(str), prefix=col, drop_first=True)
            feature_frames.append(dummies)
            continue

    if not feature_frames:
        raise TrainingError("No usable feature columns remain after encoding. Cannot train a model.")

    X = pd.concat(feature_frames, axis=1)
    X = X.loc[:, ~X.columns.duplicated()]
    # pd.get_dummies returns bool dtype columns in recent pandas versions.
    # A mixed bool/int/float matrix trains fine in sklearn/XGBoost, but
    # SHAP's TreeExplainer C extension requires pure float64 — cast here
    # once, at the source, so /train and /explain both get a clean matrix.
    X = X.astype("float64")

    # target
    target_classes = None
    if task_type == "classification":
        codes, uniques = pd.factorize(work_df[target_column].astype(str))
        y = pd.Series(codes, index=work_df.index)
        target_classes = list(uniques)
    else:
        y = pd.to_numeric(work_df[target_column], errors="coerce")
        valid_mask = y.notna()
        y = y[valid_mask]
        X = X.loc[valid_mask]

    return X, y, dropped, task_type, target_classes


def _regression_metrics(y_true, y_pred) -> ModelMetrics:
    return ModelMetrics(
        r2=round(float(r2_score(y_true, y_pred)), 4),
        rmse=round(float(root_mean_squared_error(y_true, y_pred)), 4),
    )


def _classification_metrics(y_true, y_pred) -> ModelMetrics:
    return ModelMetrics(
        accuracy=round(float(accuracy_score(y_true, y_pred)), 4),
        f1=round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        precision=round(float(precision_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
    )


def train_and_compare(
    X: pd.DataFrame,
    y: pd.Series,
    task_type: TaskType,
) -> tuple[list[ModelResult], str, object]:
    """Returns (results, best_model_name, best_fitted_model_object)."""

    stratify = y if (task_type == "classification" and y.value_counts().min() >= 2) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify
    )

    if task_type == "regression":
        candidates = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
            "XGBoost": XGBRegressor(n_estimators=200, random_state=RANDOM_STATE, verbosity=0, base_score=0.5),
        }
    else:
        candidates = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
            "XGBoost": XGBClassifier(
                n_estimators=200, random_state=RANDOM_STATE, use_label_encoder=False,
                eval_metric="logloss", verbosity=0, base_score=0.5,
            ),
        }

    results: list[ModelResult] = []
    fitted_models: dict[str, object] = {}

    for name, model in candidates.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = (
            _regression_metrics(y_test, preds)
            if task_type == "regression"
            else _classification_metrics(y_test, preds)
        )
        results.append(ModelResult(model_name=name, metrics=metrics, is_best=False))
        fitted_models[name] = model

    # pick best: higher R2 for regression, higher F1 for classification
    if task_type == "regression":
        best = max(results, key=lambda r: r.metrics.r2)
    else:
        best = max(results, key=lambda r: r.metrics.f1)

    for r in results:
        r.is_best = r.model_name == best.model_name

    return results, best.model_name, fitted_models[best.model_name]