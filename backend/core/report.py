"""
/report - assembles a single downloadable PDF combining everything the
other endpoints already compute: dataset overview, key charts, AI
insight paragraph, and (if /train has run) the model comparison table
and top SHAP features. Nothing new is computed here except the charts
themselves — this module is presentation, not analysis.
"""
from __future__ import annotations
import io
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # headless — no display server available on a server
import matplotlib.pyplot as plt
import pandas as pd
from fpdf import FPDF

from core.cleaning import build_column_info
from core.eda import (
    compute_numeric_summary,
    compute_categorical_summary,
    compute_correlation,
    compute_missing_summary,
)
from core.insight import generate_insight
from core.session_store import Session

MAX_HISTOGRAMS = 3
MAX_BAR_CHARTS = 2
PAGE_WIDTH_MM = 190  # usable width on an A4 page with default margins


def _fig_to_buffer(fig) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def _make_correlation_heatmap(df: pd.DataFrame, numeric_cols: list[str]):
    if len(numeric_cols) < 2:
        return None
    corr = df[numeric_cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title("Correlation Matrix", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return _fig_to_buffer(fig)


def _make_histogram(df: pd.DataFrame, column: str):
    series = df[column].dropna()
    if len(series) < 2 or series.nunique() < 2:
        return None
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(series, bins=15, color="#4C72B0", edgecolor="white")
    ax.set_title(f"Distribution: {column}", fontsize=10)
    ax.set_xlabel(column, fontsize=8)
    ax.set_ylabel("Count", fontsize=8)
    fig.tight_layout()
    return _fig_to_buffer(fig)


def _make_bar_chart(df: pd.DataFrame, column: str):
    counts = df[column].dropna().astype(str).value_counts().head(10)
    if len(counts) == 0:
        return None
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(counts.index.astype(str), counts.values, color="#55A868")
    ax.set_title(f"Top values: {column}", fontsize=10)
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.set_ylabel("Count", fontsize=8)
    fig.tight_layout()
    return _fig_to_buffer(fig)


def _sanitize_text(text: str) -> str:
    """
    fpdf2's core fonts (Helvetica etc.) only support latin-1, not full
    unicode. LLM-generated text (insight paragraphs, chat) can contain
    em-dashes, smart quotes, bullets, etc. that crash pdf.cell()/multi_cell()
    outright rather than degrading gracefully. Replace the common offenders
    with ASCII equivalents, then fall back to '?' for anything else rather
    than raising FPDFUnicodeEncodingException mid-report.
    """
    replacements = {
        "\u2014": "-", "\u2013": "-",   # em dash, en dash
        "\u2018": "'", "\u2019": "'",   # smart single quotes
        "\u201c": '"', "\u201d": '"',   # smart double quotes
        "\u2022": "-", "\u2026": "...",  # bullet, ellipsis
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text.encode("latin-1", "replace").decode("latin-1")


class ReportPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, _sanitize_text("AI Data Analyst - Report"), ln=True)
        self.set_font("Helvetica", "", 9)
        self.set_text_color(120, 120, 120)
        self.cell(0, 6, datetime.now().strftime("Generated %Y-%m-%d %H:%M"), ln=True)
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def section_title(self, text: str):
        self.set_font("Helvetica", "B", 12)
        self.set_fill_color(230, 236, 245)
        self.cell(0, 8, _sanitize_text(text), ln=True, fill=True)
        self.ln(2)

    def body_text(self, text: str):
        self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 5.5, _sanitize_text(text))
        self.ln(1)


def generate_report_pdf(session: Session) -> bytes:
    df = session.df
    columns = build_column_info(df)
    numeric_cols = [c.name for c in columns if c.inferred_type == "numeric"]
    categorical_cols = [c.name for c in columns if c.inferred_type == "categorical"]

    numeric_summary = compute_numeric_summary(df, numeric_cols)
    categorical_summary = compute_categorical_summary(df, categorical_cols)
    missing_summary = compute_missing_summary(columns)

    insight_text, _ = generate_insight(
        n_rows=df.shape[0],
        n_columns=df.shape[1],
        numeric_summary=numeric_summary,
        categorical_summary=categorical_summary,
        missing_summary=missing_summary,
    )

    pdf = ReportPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # --- Dataset overview ---
    pdf.section_title("Dataset Overview")
    pdf.body_text(
        f"File: {session.filename}\n"
        f"Rows: {df.shape[0]}   Columns: {df.shape[1]}\n"
        f"Column types: {len(numeric_cols)} numeric, {len(categorical_cols)} categorical, "
        f"{sum(1 for c in columns if c.inferred_type == 'datetime')} datetime, "
        f"{sum(1 for c in columns if c.inferred_type == 'boolean')} boolean, "
        f"{sum(1 for c in columns if c.inferred_type == 'text')} text"
    )

    if missing_summary:
        missing_lines = "\n".join(
            f"  - {m.column}: {m.missing_count} missing ({m.missing_pct}%)" for m in missing_summary
        )
        pdf.body_text(f"Columns with missing values:\n{missing_lines}")
    else:
        pdf.body_text("No missing values in this dataset.")

    # --- AI insight paragraph ---
    pdf.section_title("AI-Generated Insight")
    pdf.body_text(insight_text)

    # --- Charts ---
    pdf.section_title("Key Charts")

    heatmap_buf = _make_correlation_heatmap(df, numeric_cols)
    if heatmap_buf is not None:
        pdf.image(heatmap_buf, w=PAGE_WIDTH_MM * 0.7)
        pdf.ln(3)

    for col in numeric_cols[:MAX_HISTOGRAMS]:
        buf = _make_histogram(df, col)
        if buf is not None:
            pdf.image(buf, w=PAGE_WIDTH_MM * 0.7)
            pdf.ln(3)

    for col in categorical_cols[:MAX_BAR_CHARTS]:
        buf = _make_bar_chart(df, col)
        if buf is not None:
            pdf.image(buf, w=PAGE_WIDTH_MM * 0.7)
            pdf.ln(3)

    # --- Model results (only if /train has run on this session) ---
    if session.trained_model is not None and session.last_train_results:
        pdf.add_page()
        pdf.section_title("Model Comparison")
        pdf.body_text(
            f"Task type: {session.task_type}\n"
            f"Target column: {session.target_column}\n"
            f"Best model: {session.model_name}"
        )

        for result in session.last_train_results:
            metrics = result["metrics"]
            metric_line = ", ".join(f"{k}={v}" for k, v in metrics.items() if v is not None)
            marker = " (BEST)" if result["is_best"] else ""
            pdf.body_text(f"- {result['model_name']}{marker}: {metric_line}")

        if session.dropped_columns:
            dropped_lines = "\n".join(
                f"  - {d['column']}: {d['reason']}" for d in session.dropped_columns
            )
            pdf.body_text(f"Columns excluded from training:\n{dropped_lines}")

    return bytes(pdf.output())