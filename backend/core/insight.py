"""
Generates the one-paragraph plain-English insight for the EDA response.

Grounding principle (per blueprint): the LLM never sees the raw dataset
and never invents numbers — it only receives a compact JSON summary of
stats we've already computed server-side, and is asked to narrate it.
This is the same pattern as ResumeFit's AI feedback layer.

If GROQ_API_KEY isn't set (e.g. local dev without a key yet), this
degrades gracefully: EDAResponse.insight_generated = False and a
placeholder message, rather than crashing the whole /eda call.
"""
from __future__ import annotations
import os
import json
import requests

from core.schemas import NumericSummary, CategoricalSummary, MissingSummaryItem

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"
REQUEST_TIMEOUT_SECONDS = 15

FALLBACK_MESSAGE = (
    "AI insight generation is unavailable right now (no GROQ_API_KEY configured). "
    "The stats above are still fully computed — this paragraph is just the "
    "LLM-written narration layer on top of them."
)


def _build_prompt(
    n_rows: int,
    n_columns: int,
    numeric_summary: list[NumericSummary],
    categorical_summary: list[CategoricalSummary],
    missing_summary: list[MissingSummaryItem],
) -> str:
    compact_stats = {
        "n_rows": n_rows,
        "n_columns": n_columns,
        "numeric_columns": [
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
        "categorical_columns": [
            {
                "column": s.column,
                "top_value": s.top_value,
                "top_value_count": s.top_value_count,
                "n_unique": s.n_unique,
            }
            for s in categorical_summary
        ],
        "columns_with_missing_values": [
            {"column": m.column, "missing_pct": m.missing_pct}
            for m in missing_summary
        ],
    }

    return (
        "You are a data analyst writing a short plain-English summary of a dataset "
        "for a non-technical reader. You are given precomputed statistics as JSON — "
        "use ONLY these numbers, never invent or estimate values not present here. "
        "Write exactly 3-5 sentences covering: what stands out, any notable "
        "distributions or imbalances, and any data quality issues (missing values). "
        "Do not use markdown formatting, headers, or bullet points — plain prose only.\n\n"
        f"Stats:\n{json.dumps(compact_stats, indent=2)}"
    )


def generate_insight(
    n_rows: int,
    n_columns: int,
    numeric_summary: list[NumericSummary],
    categorical_summary: list[CategoricalSummary],
    missing_summary: list[MissingSummaryItem],
) -> tuple[str, bool]:
    """Returns (insight_text, was_generated_by_llm)."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return FALLBACK_MESSAGE, False

    prompt = _build_prompt(n_rows, n_columns, numeric_summary, categorical_summary, missing_summary)

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
                "temperature": 0.4,
                "max_tokens": 300,
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()
        text = data["choices"][0]["message"]["content"].strip()
        return text, True
    except Exception as e:
        return f"AI insight generation failed: {e}", False