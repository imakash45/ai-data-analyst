"""
/train - user picks a target column, we auto-detect regression vs
         classification, encode features deterministically, train
         all 3 candidate models, and return a comparison table with
         the auto-selected best model. The winning fitted model is
         cached on the session for later reuse (/explain, /chat).
"""
from __future__ import annotations
from fastapi import APIRouter, HTTPException

from core.schemas import TrainRequest, TrainResponse
from core.session_store import store
from core.cleaning import build_column_info
from core.ml import prepare_features, train_and_compare, TrainingError

router = APIRouter(tags=["ml"])


@router.post("/train", response_model=TrainResponse)
async def train_models(request: TrainRequest):
    session = store.get(request.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found. Upload a file first.")

    df = session.df
    columns = build_column_info(df)

    try:
        X, y, dropped, task_type, target_classes = prepare_features(
            df, request.target_column, columns, excluded_columns=request.excluded_columns
        )
    except TrainingError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if len(X) < 20:
        raise HTTPException(
            status_code=400,
            detail=f"Only {len(X)} usable rows after dropping missing targets — too few to train reliably.",
        )

    results, best_model_name, best_model = train_and_compare(X, y, task_type)

    store.update_training(
        session_id=request.session_id,
        trained_model=best_model,
        model_name=best_model_name,
        task_type=task_type,
        target_column=request.target_column,
        feature_columns=list(X.columns),
        dropped_columns=[d.model_dump() for d in dropped],
        target_classes=target_classes,
        last_train_results=[r.model_dump() for r in results],
    )

    return TrainResponse(
        session_id=request.session_id,
        task_type=task_type,
        target_column=request.target_column,
        n_rows_used=len(X),
        n_features_used=X.shape[1],
        dropped_columns=dropped,
        results=results,
        best_model=best_model_name,
    )