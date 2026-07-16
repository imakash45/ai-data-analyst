"""
/upload  - accept a CSV or Excel file, validate it, infer column types,
           store it in the session store, return a profile.

/clean   - apply user-chosen cleaning rules (imputation, duplicate removal,
           dtype overrides) to an existing session's DataFrame.
"""
from __future__ import annotations
import io

import pandas as pd
from fastapi import APIRouter, UploadFile, File, HTTPException

from core.schemas import (
    UploadResponse,
    CleaningRequest,
    CleaningResponse,
)
from core.cleaning import (
    build_column_info,
    count_duplicate_rows,
    detect_outliers_iqr,
    apply_imputation,
    apply_dtype_override,
)
from core.session_store import store

router = APIRouter(tags=["ingestion"])

MAX_FILE_SIZE_MB = 25
ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


def _read_dataframe(filename: str, raw_bytes: bytes) -> pd.DataFrame:
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    buffer = io.BytesIO(raw_bytes)
    try:
        if ext == ".csv":
            df = pd.read_csv(buffer)
        elif ext == ".xls":
            df = pd.read_excel(buffer, engine="xlrd")
        else:
            df = pd.read_excel(buffer, engine="openpyxl")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse file: {e}")

    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded file has no rows.")
    if df.shape[1] == 0:
        raise HTTPException(status_code=400, detail="Uploaded file has no columns.")

    return df


@router.post("/upload", response_model=UploadResponse)
async def upload_dataset(file: UploadFile = File(...)):
    raw_bytes = await file.read()

    size_mb = len(raw_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({size_mb:.1f} MB). Limit is {MAX_FILE_SIZE_MB} MB.",
        )
    if size_mb == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    df = _read_dataframe(file.filename, raw_bytes)

    session = store.create(filename=file.filename, df=df)

    warnings: list[str] = []
    if df.shape[0] < 20:
        warnings.append("Dataset has fewer than 20 rows — EDA/ML results may be unreliable.")
    if df.shape[1] > 200:
        warnings.append("Dataset has a very large number of columns — some views may be truncated.")

    return UploadResponse(
        session_id=session.session_id,
        filename=file.filename,
        n_rows=df.shape[0],
        n_columns=df.shape[1],
        columns=build_column_info(df),
        duplicate_row_count=count_duplicate_rows(df),
        warnings=warnings,
    )


@router.post("/clean", response_model=CleaningResponse)
async def clean_dataset(request: CleaningRequest):
    session = store.get(request.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found. Upload a file first.")

    df = session.df.copy()
    n_before = len(df)

    # 1. dtype overrides first, so imputation runs against corrected types
    for column, target_type in request.dtype_overrides.items():
        df = apply_dtype_override(df, column, target_type)

    # 2. duplicate removal
    rows_dropped_dupes = 0
    if request.drop_duplicates:
        n_pre_dupe = len(df)
        df = df.drop_duplicates()
        rows_dropped_dupes = n_pre_dupe - len(df)

    # 3. per-column imputation rules
    for rule in request.rules:
        df = apply_imputation(df, rule)

    n_after = len(df)

    numeric_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
    ]
    outliers = detect_outliers_iqr(df, numeric_cols)

    store.update_df(request.session_id, df)

    return CleaningResponse(
        session_id=request.session_id,
        n_rows_before=n_before,
        n_rows_after=n_after,
        rows_dropped=n_before - n_after,
        applied_rules=request.rules,
        outliers=outliers,
        columns=build_column_info(df),
    )
