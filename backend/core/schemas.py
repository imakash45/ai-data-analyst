"""
Pydantic schemas shared across all routers.
Keeping these in one place means /eda, /train, /explain, /chat, /report
all speak the same "shape" of data — no duplicated request/response models.
"""
from __future__ import annotations
from typing import Optional, Literal
from pydantic import BaseModel, Field, ConfigDict


# ---------- Column typing ----------

ColumnType = Literal["numeric", "categorical", "datetime", "text", "boolean"]


class ColumnInfo(BaseModel):
    name: str
    dtype: str                 # raw pandas dtype as string, e.g. "int64"
    inferred_type: ColumnType  # our semantic classification
    missing_count: int
    missing_pct: float
    unique_count: int
    sample_values: list


# ---------- /upload ----------

class UploadResponse(BaseModel):
    session_id: str
    filename: str
    n_rows: int
    n_columns: int
    columns: list[ColumnInfo]
    duplicate_row_count: int
    warnings: list[str] = Field(default_factory=list)


# ---------- /clean ----------

ImputationStrategy = Literal["mean", "median", "mode", "drop_rows", "leave_as_is"]


class ColumnCleaningRule(BaseModel):
    column: str
    imputation: ImputationStrategy = "leave_as_is"


class CleaningRequest(BaseModel):
    session_id: str
    rules: list[ColumnCleaningRule] = Field(default_factory=list)
    drop_duplicates: bool = False
    dtype_overrides: dict[str, ColumnType] = Field(default_factory=dict)


class OutlierFlag(BaseModel):
    column: str
    outlier_count: int
    lower_bound: float
    upper_bound: float


class CleaningResponse(BaseModel):
    session_id: str
    n_rows_before: int
    n_rows_after: int
    rows_dropped: int
    applied_rules: list[ColumnCleaningRule]
    outliers: list[OutlierFlag]
    columns: list[ColumnInfo]


# ---------- generic error ----------

class ErrorResponse(BaseModel):
    detail: str


# ---------- /eda ----------

class EDARequest(BaseModel):
    session_id: str


class NumericSummary(BaseModel):
    column: str
    count: int
    mean: float
    std: float
    min: float
    q25: float
    median: float
    q75: float
    max: float


class CategoricalSummary(BaseModel):
    column: str
    top_value: Optional[str] = None
    top_value_count: int = 0
    n_unique: int = 0
    value_counts: dict[str, int] = Field(default_factory=dict)  # top N only


class HistogramBin(BaseModel):
    bin_start: float
    bin_end: float
    count: int


class DistributionData(BaseModel):
    column: str
    bins: list[HistogramBin]


class CorrelationMatrix(BaseModel):
    columns: list[str]
    matrix: list[list[float]]  # square matrix, same order as `columns`


class MissingSummaryItem(BaseModel):
    column: str
    missing_count: int
    missing_pct: float


class EDAResponse(BaseModel):
    session_id: str
    n_rows: int
    n_columns: int
    numeric_summary: list[NumericSummary]
    categorical_summary: list[CategoricalSummary]
    distributions: list[DistributionData]
    correlation: Optional[CorrelationMatrix] = None
    missing_summary: list[MissingSummaryItem]
    insight_text: str
    insight_generated: bool  # False if LLM call was skipped/failed (e.g. no API key)
    
# ---------- /train ----------

TaskType = Literal["regression", "classification"]


class TrainRequest(BaseModel):
    session_id: str
    target_column: str
    excluded_columns: list[str] = []


class DroppedColumn(BaseModel):
    column: str
    reason: str


class ModelMetrics(BaseModel):
    # Regression fields (None for classification results)
    r2: Optional[float] = None
    rmse: Optional[float] = None
    # Classification fields (None for regression results)
    accuracy: Optional[float] = None
    f1: Optional[float] = None
    precision: Optional[float] = None


class ModelResult(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    metrics: ModelMetrics
    is_best: bool


class TrainResponse(BaseModel):
    session_id: str
    task_type: TaskType
    target_column: str
    n_rows_used: int
    n_features_used: int
    dropped_columns: list[DroppedColumn]
    results: list[ModelResult]
    best_model: str
    
# ---------- /explain ----------

class ExplainRequest(BaseModel):
    session_id: str


class FeatureImportance(BaseModel):
    feature: str
    importance: float  # mean |SHAP value| across sampled rows — higher = more influential


class ExplainResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    session_id: str
    model_name: str
    task_type: TaskType
    target_column: str
    feature_importances: list[FeatureImportance]  # sorted descending by importance
    n_rows_sampled: int
    
# ---------- /chat ----------

class ChatRequest(BaseModel):
    session_id: str
    question: str


class ChatResponse(BaseModel):
    session_id: str
    question: str
    answer: str
    answer_generated: bool  # False if LLM call was skipped/failed (e.g. no API key)
    
# ---------- /report ----------

class ReportRequest(BaseModel):
    session_id: str