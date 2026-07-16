"""
/explain - runs SHAP on the model /train already fit and cached on this
           session. Rebuilds the same encoded feature matrix /train used
           (same deterministic encoding rule in core/ml.py) rather than
           storing the full matrix on the session, to keep memory light.
"""
from __future__ import annotations
from fastapi import APIRouter, HTTPException

from core.schemas import ExplainRequest, ExplainResponse, FeatureImportance
from core.session_store import store
from core.cleaning import build_column_info
from core.ml import prepare_features, TrainingError
from core.explain import compute_feature_importance

router = APIRouter(tags=["explain"])


@router.post("/explain", response_model=ExplainResponse)
async def explain_model(request: ExplainRequest):
    session = store.get(request.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found. Upload a file first.")

    if session.trained_model is None:
        raise HTTPException(
            status_code=400,
            detail="No trained model on this session yet. Call /train first.",
        )

    df = session.df
    columns = build_column_info(df)

    try:
        X, y, dropped, task_type, target_classes = prepare_features(df, session.target_column, columns)
    except TrainingError as e:
        raise HTTPException(status_code=400, detail=f"Could not rebuild features for explanation: {e}")

    # Align to the exact column set/order the model was trained on.
    # (Deterministic encoding means this should already match, but this
    # guards against any drift — e.g. a category that dropped out of a sample.)
    X = X.reindex(columns=session.feature_columns, fill_value=0)

    pairs, n_sampled = compute_feature_importance(session.trained_model, X)

    return ExplainResponse(
        session_id=request.session_id,
        model_name=session.model_name,
        task_type=session.task_type,
        target_column=session.target_column,
        feature_importances=[FeatureImportance(feature=f, importance=round(v, 6)) for f, v in pairs],
        n_rows_sampled=n_sampled,
    )