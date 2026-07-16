"""
In-memory session store.

Deliberately simple: a single-process dict keyed by session_id (uuid4).
No Redis, no DB — this is out of scope per the project blueprint.
Good enough for a demo/portfolio deployment (single Render instance).

Each session holds:
    - df: the current pandas DataFrame (mutated in place as cleaning is applied)
    - filename: original upload filename
    - created_at: for optional future TTL cleanup
    - trained_model / feature_columns / etc: populated by /train (Module 3),
      reused later by /explain (SHAP) and /chat so those endpoints don't
      need to retrain from scratch.
"""
from __future__ import annotations
import uuid
import time
from dataclasses import dataclass, field
from typing import Any, Optional
import pandas as pd


@dataclass
class Session:
    session_id: str
    filename: str
    df: pd.DataFrame
    created_at: float = field(default_factory=time.time)

    # Populated after /train — None until then.
    trained_model: Optional[Any] = None
    model_name: Optional[str] = None            # e.g. "XGBoost"
    task_type: Optional[str] = None             # "regression" | "classification"
    target_column: Optional[str] = None
    feature_columns: Optional[list[str]] = None  # post-encoding column order (must match at predict time)
    dropped_columns: Optional[list[dict]] = None  # [{"column": ..., "reason": ...}, ...]
    target_classes: Optional[list] = None        # for classification: original label order behind encoded ints
    last_train_results: Optional[list[dict]] = None  # full model comparison table, for /report reuse


class SessionStore:
    def __init__(self):
        self._sessions: dict[str, Session] = {}

    def create(self, filename: str, df: pd.DataFrame) -> Session:
        session_id = str(uuid.uuid4())
        session = Session(session_id=session_id, filename=filename, df=df)
        self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def update_df(self, session_id: str, df: pd.DataFrame) -> None:
        if session_id in self._sessions:
            self._sessions[session_id].df = df

    def update_training(
        self,
        session_id: str,
        trained_model: Any,
        model_name: str,
        task_type: str,
        target_column: str,
        feature_columns: list[str],
        dropped_columns: list[dict],
        target_classes: Optional[list] = None,
        last_train_results: Optional[list[dict]] = None,
    ) -> None:
        session = self._sessions.get(session_id)
        if session is None:
            return
        session.trained_model = trained_model
        session.model_name = model_name
        session.task_type = task_type
        session.target_column = target_column
        session.feature_columns = feature_columns
        session.dropped_columns = dropped_columns
        session.target_classes = target_classes
        session.last_train_results = last_train_results

    def exists(self, session_id: str) -> bool:
        return session_id in self._sessions


# Single shared instance imported by routers.
# (FastAPI runs as one process for this project, so a module-level
# singleton is sufficient — no need for dependency-injected state.)
store = SessionStore()