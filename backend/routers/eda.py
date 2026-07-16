"""
/eda - runs the full auto-EDA pipeline against a session's current
        (possibly already-cleaned) DataFrame and returns everything
        the frontend needs to render the EDA page in one call.
"""
from __future__ import annotations
import pandas as pd
from fastapi import APIRouter, HTTPException

from core.schemas import EDARequest, EDAResponse
from core.session_store import store
from core.cleaning import build_column_info
from core.eda import (
    compute_numeric_summary,
    compute_categorical_summary,
    compute_distributions,
    compute_correlation,
    compute_missing_summary,
)
from core.insight import generate_insight

router = APIRouter(tags=["eda"])


@router.post("/eda", response_model=EDAResponse)
async def run_eda(request: EDARequest):
    session = store.get(request.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found. Upload a file first.")

    df = session.df
    columns = build_column_info(df)

    numeric_cols = [c.name for c in columns if c.inferred_type == "numeric"]
    categorical_cols = [c.name for c in columns if c.inferred_type == "categorical"]

    numeric_summary = compute_numeric_summary(df, numeric_cols)
    categorical_summary = compute_categorical_summary(df, categorical_cols)
    distributions = compute_distributions(df, numeric_cols)
    correlation = compute_correlation(df, numeric_cols)
    missing_summary = compute_missing_summary(columns)

    insight_text, insight_generated = generate_insight(
        n_rows=df.shape[0],
        n_columns=df.shape[1],
        numeric_summary=numeric_summary,
        categorical_summary=categorical_summary,
        missing_summary=missing_summary,
    )

    return EDAResponse(
        session_id=request.session_id,
        n_rows=df.shape[0],
        n_columns=df.shape[1],
        numeric_summary=numeric_summary,
        categorical_summary=categorical_summary,
        distributions=distributions,
        correlation=correlation,
        missing_summary=missing_summary,
        insight_text=insight_text,
        insight_generated=insight_generated,
    )