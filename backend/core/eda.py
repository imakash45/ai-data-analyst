"""
EDA computation — pure pandas/numpy, no LLM calls here (that lives in
core/insight.py). Every function takes a DataFrame + column type info
and returns plain data; the router assembles these into EDAResponse.
"""
from __future__ import annotations
import math
import pandas as pd
import numpy as np

from core.schemas import (
    NumericSummary,
    CategoricalSummary,
    HistogramBin,
    DistributionData,
    CorrelationMatrix,
    MissingSummaryItem,
    ColumnInfo,
)

TOP_N_CATEGORIES = 10
HISTOGRAM_BINS = 15


def _safe_float(value) -> float:
    """NaN/inf can't serialize to JSON — coerce to 0.0 so responses never break."""
    if value is None:
        return 0.0
    f = float(value)
    if math.isnan(f) or math.isinf(f):
        return 0.0
    return f


def compute_numeric_summary(df: pd.DataFrame, numeric_columns: list[str]) -> list[NumericSummary]:
    summaries = []
    for col in numeric_columns:
        series = df[col].dropna()
        if len(series) == 0:
            continue
        summaries.append(
            NumericSummary(
                column=col,
                count=int(len(series)),
                mean=_safe_float(series.mean()),
                std=_safe_float(series.std()),
                min=_safe_float(series.min()),
                q25=_safe_float(series.quantile(0.25)),
                median=_safe_float(series.median()),
                q75=_safe_float(series.quantile(0.75)),
                max=_safe_float(series.max()),
            )
        )
    return summaries


def compute_categorical_summary(df: pd.DataFrame, categorical_columns: list[str]) -> list[CategoricalSummary]:
    summaries = []
    for col in categorical_columns:
        series = df[col].dropna().astype(str)
        if len(series) == 0:
            summaries.append(CategoricalSummary(column=col))
            continue
        counts = series.value_counts().head(TOP_N_CATEGORIES)
        summaries.append(
            CategoricalSummary(
                column=col,
                top_value=str(counts.index[0]) if len(counts) > 0 else None,
                top_value_count=int(counts.iloc[0]) if len(counts) > 0 else 0,
                n_unique=int(series.nunique()),
                value_counts={str(k): int(v) for k, v in counts.items()},
            )
        )
    return summaries


def compute_distributions(df: pd.DataFrame, numeric_columns: list[str]) -> list[DistributionData]:
    distributions = []
    for col in numeric_columns:
        series = df[col].dropna()
        if len(series) < 2 or series.nunique() < 2:
            continue
        counts, bin_edges = np.histogram(series, bins=HISTOGRAM_BINS)
        bins = [
            HistogramBin(
                bin_start=_safe_float(bin_edges[i]),
                bin_end=_safe_float(bin_edges[i + 1]),
                count=int(counts[i]),
            )
            for i in range(len(counts))
        ]
        distributions.append(DistributionData(column=col, bins=bins))
    return distributions


def compute_correlation(df: pd.DataFrame, numeric_columns: list[str]) -> CorrelationMatrix | None:
    if len(numeric_columns) < 2:
        return None
    corr = df[numeric_columns].corr(numeric_only=True)
    corr = corr.fillna(0.0)
    matrix = [[_safe_float(v) for v in row] for row in corr.values]
    return CorrelationMatrix(columns=list(corr.columns), matrix=matrix)


def compute_missing_summary(columns: list[ColumnInfo]) -> list[MissingSummaryItem]:
    return [
        MissingSummaryItem(
            column=c.name,
            missing_count=c.missing_count,
            missing_pct=c.missing_pct,
        )
        for c in columns
        if c.missing_count > 0
    ]