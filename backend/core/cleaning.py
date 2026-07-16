"""
Data cleaning primitives.

Design principle (per blueprint): cleaning is automatic but SHOWN to the
user, never silently forced. Every function here either:
  (a) inspects and reports (infer_column_types, detect_outliers_iqr), or
  (b) applies a rule the user explicitly chose (apply_imputation, drop_duplicates)

Nothing in this module guesses on the user's behalf and hides the guess.
"""
from __future__ import annotations
import pandas as pd
import numpy as np

from core.schemas import ColumnInfo, ColumnType, ColumnCleaningRule, OutlierFlag


MAX_CATEGORICAL_UNIQUE_RATIO = 0.5   # if unique/total <= this, treat as categorical
MAX_CATEGORICAL_UNIQUE_ABS = 50      # ...unless there are too many distinct values
SAMPLE_SIZE = 5


def infer_column_type(series: pd.Series) -> ColumnType:
    """Classify a column into one semantic bucket used by the rest of the pipeline."""
    if pd.api.types.is_bool_dtype(series):
        return "boolean"

    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    # try datetime parse for object columns that look like dates
    if series.dtype == object:
        non_null = series.dropna()
        if len(non_null) > 0:
            parsed = pd.to_datetime(non_null, errors="coerce", format="mixed")
            if parsed.notna().mean() > 0.9:
                return "datetime"

    if pd.api.types.is_numeric_dtype(series):
        n_unique = series.nunique(dropna=True)
        # only treat as categorical if effectively binary/flag-like (e.g. 0/1)
        if n_unique <= 2:
            return "categorical"
        return "numeric"

    # object / string column
    n_unique = series.nunique(dropna=True)
    n_total = max(len(series), 1)
    if n_unique / n_total <= MAX_CATEGORICAL_UNIQUE_RATIO and n_unique <= MAX_CATEGORICAL_UNIQUE_ABS:
        return "categorical"
    return "text"


def build_column_info(df: pd.DataFrame) -> list[ColumnInfo]:
    infos = []
    n_rows = len(df)
    for col in df.columns:
        series = df[col]
        missing = int(series.isna().sum())
        infos.append(
            ColumnInfo(
                name=col,
                dtype=str(series.dtype),
                inferred_type=infer_column_type(series),
                missing_count=missing,
                missing_pct=round(100 * missing / n_rows, 2) if n_rows else 0.0,
                unique_count=int(series.nunique(dropna=True)),
                sample_values=series.dropna().head(SAMPLE_SIZE).tolist(),
            )
        )
    return infos


def count_duplicate_rows(df: pd.DataFrame) -> int:
    return int(df.duplicated().sum())


def detect_outliers_iqr(df: pd.DataFrame, numeric_columns: list[str]) -> list[OutlierFlag]:
    """IQR method. Flags only — never removes automatically."""
    flags = []
    for col in numeric_columns:
        series = df[col].dropna()
        if len(series) < 4:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_count = int(((series < lower) | (series > upper)).sum())
        if outlier_count > 0:
            flags.append(
                OutlierFlag(
                    column=col,
                    outlier_count=outlier_count,
                    lower_bound=round(float(lower), 4),
                    upper_bound=round(float(upper), 4),
                )
            )
    return flags


def apply_imputation(df: pd.DataFrame, rule: ColumnCleaningRule) -> pd.DataFrame:
    col = rule.column
    if col not in df.columns:
        return df

    strategy = rule.imputation
    if strategy == "leave_as_is":
        return df

    if strategy == "drop_rows":
        return df.dropna(subset=[col])

    is_true_numeric = pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col])

    if strategy == "mean":
        if is_true_numeric:
            df[col] = df[col].fillna(df[col].mean())
        return df

    if strategy == "median":
        if is_true_numeric:
            df[col] = df[col].fillna(df[col].median())
        return df

    if strategy == "mode":
        mode_vals = df[col].mode(dropna=True)
        if len(mode_vals) > 0:
            df[col] = df[col].fillna(mode_vals.iloc[0])
        return df

    return df


def apply_dtype_override(df: pd.DataFrame, column: str, target_type: ColumnType) -> pd.DataFrame:
    if column not in df.columns:
        return df
    try:
        if target_type == "numeric":
            df[column] = pd.to_numeric(df[column], errors="coerce")
        elif target_type == "datetime":
            df[column] = pd.to_datetime(df[column], errors="coerce", format="mixed")
        elif target_type == "categorical":
            df[column] = df[column].astype("category")
        elif target_type == "boolean":
            df[column] = df[column].astype(bool)
        # "text" -> leave as object, no-op
    except Exception:
        # If the coercion fails outright, leave the column untouched
        # rather than raising — this is a best-effort suggestion, not a hard rule.
        pass
    return df
