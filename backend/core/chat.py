"""
/chat pipeline logic.

Grounding principle (same as core/insight.py): the LLM never sees the raw
dataset. It only receives a compact JSON context built from stats we've
already computed server-side (reusing core/eda.py + core/cleaning.py),
plus /train and /explain results if the session has them. This keeps
answers tied to real numbers instead of the model guessing/hallucinating
over a dataset it never actually saw.
"""
from __future__ import annotations
import os
import json
import requests
import pandas as pd

from core.cleaning import build_column_info
from core.eda import (
    compute_numeric_summary,
    compute_categorical_summary,
    compute_correlation,
    compute_missing_summary,
)
from core.session_store import Session

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"
REQUEST_TIMEOUT_SECONDS = 15

FALLBACK_MESSAGE = (
    "AI chat is unavailable right now (no GROQ_API_KEY configured). "
    "Once a key is set, this endpoint answers questions grounded in the "
    "same stats /eda, /train, and /explain already compute."
)


def build_dataset_context(df: pd.DataFrame, session: Session) -> dict:
    """Assemble everything the LLM is allowed to know about this dataset."""
    columns = build_column_info(df)
    numeric_cols = [c.name for c in columns if c.inferred_type == "numeric"]
    categorical_cols = [c.name for c in columns if c.inferred_type == "categorical"]

    numeric_summary = compute_numeric_summary(df, numeric_cols)
    categorical_summary = compute_categorical_summary(df, categorical_cols)
    correlation = compute_correlation(df, numeric_cols)
    missing_summary = compute_missing_summary(columns)

    context = {
        "n_rows": df.shape[0],
        "n_columns": df.shape[1],
        "columns": [
            {
                "name": c.name,
                "type": c.inferred_type,
                "missing_count": c.missing_count,
                "missing_pct": c.missing_pct,
                "unique_count": c.unique_count,
            }
            for c in columns
        ],
        "numeric_summary": [
            {
                "column": s.column,
                "mean": round(s.mean, 2),
                "median": round(s.median, 2),
                "min": round(s.min, 2),
                "max": round(s.max, 2),
                "std": round(s.std, 2),
            }
            for s in numeric_summary
        ],
        "categorical_summary": [
            {
                "column": s.column,
                "top_value": s.top_value,
                "top_value_count": s.top_value_count,
                "n_unique": s.n_unique,
                "value_counts": s.value_counts,
            }
            for s in categorical_summary
        ],
        "missing_summary": [
            {"column": m.column, "missing_count": m.missing_count, "missing_pct": m.missing_pct}
            for m in missing_summary
        ],
        "correlation": (
            {"columns": correlation.columns, "matrix": correlation.matrix}
            if correlation is not None
            else None
        ),
    }

    # If /train has already run on this session, include the model results too
    # so questions like "what's driving the target most" can be answered.
    if session.trained_model is not None:
        context["trained_model"] = {
            "task_type": session.task_type,
            "target_column": session.target_column,
            "best_model": session.model_name,
            "dropped_columns": session.dropped_columns,
        }

    return context


def _build_prompt(question: str, context: dict) -> str:
    return (
        "You are a data analyst answering a question about a dataset for a "
        "non-technical reader. You are given precomputed statistics as JSON — "
        "use ONLY these numbers, never invent or estimate values not present here. "
        "If the question asks something these stats cannot answer, say so plainly "
        "rather than guessing. Answer in 1-4 sentences, plain prose, no markdown.\n\n"
        f"Dataset stats:\n{json.dumps(context, indent=2)}\n\n"
        f"Question: {question}"
    )


def generate_chat_answer(question: str, context: dict) -> tuple[str, bool]:
    """Returns (answer_text, was_generated_by_llm)."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return FALLBACK_MESSAGE, False

    prompt = _build_prompt(question, context)

    try:
        response = requests.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 250,
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()
        text = data["choices"][0]["message"]["content"].strip()
        return text, True
    except Exception as e:
        return f"AI chat failed: {e}", False