"""
Custom CSS injection for the Streamlit frontend, giving it a colorful,
modern "SaaS product" look instead of default Streamlit styling.
Call inject_custom_css() once at the top of app.py, right after
st.set_page_config().
"""
import streamlit as st

CUSTOM_CSS = """
<style>
/* ---------- Page background ---------- */
.stApp {
    background: linear-gradient(180deg, #f8f9ff 0%, #f0f2ff 100%);
}

/* ---------- Hero title ---------- */
.hero-title {
    background: linear-gradient(90deg, #6366f1, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 2.6rem;
    font-weight: 800;
    margin-bottom: 0;
}
.hero-caption {
    color: #6b7280;
    font-size: 1.05rem;
    margin-top: 0.2rem;
    margin-bottom: 1.5rem;
}

/* ---------- Section card wrapper ---------- */
.section-card {
    background: white;
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.4rem;
    box-shadow: 0 2px 12px rgba(99, 102, 241, 0.08);
    border: 1px solid rgba(99, 102, 241, 0.08);
}

.section-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.02em;
    color: white;
    margin-bottom: 0.6rem;
}
.badge-upload   { background: linear-gradient(90deg, #3b82f6, #60a5fa); }
.badge-clean    { background: linear-gradient(90deg, #8b5cf6, #a78bfa); }
.badge-eda      { background: linear-gradient(90deg, #14b8a6, #2dd4bf); }
.badge-train    { background: linear-gradient(90deg, #f97316, #fb923c); }
.badge-explain  { background: linear-gradient(90deg, #ec4899, #f472b6); }
.badge-chat     { background: linear-gradient(90deg, #22c55e, #4ade80); }
.badge-report   { background: linear-gradient(90deg, #6366f1, #818cf8); }

.section-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: #1f2937;
    margin-bottom: 0.15rem;
}
.section-caption {
    color: #6b7280;
    font-size: 0.92rem;
    margin-bottom: 1rem;
}

/* ---------- Buttons ---------- */
.stButton > button {
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.55rem 1.3rem;
    font-weight: 600;
    box-shadow: 0 2px 8px rgba(99, 102, 241, 0.25);
    transition: all 0.15s ease;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 14px rgba(99, 102, 241, 0.35);
    color: white;
}
.stDownloadButton > button {
    background: linear-gradient(90deg, #22c55e, #16a34a);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
}

/* ---------- Metrics ---------- */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #f5f3ff, #ede9fe);
    border-radius: 12px;
    padding: 0.8rem 1rem;
    border: 1px solid rgba(139, 92, 246, 0.15);
}

/* ---------- Dataframes ---------- */
div[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid rgba(99, 102, 241, 0.1);
}

/* ---------- Native container-as-card (st.container(border=True)) ---------- */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 16px !important;
    box-shadow: 0 2px 12px rgba(99, 102, 241, 0.08);
    border: 1px solid rgba(99, 102, 241, 0.08) !important;
    padding: 0.4rem;
}
</style>
"""


def inject_custom_css():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def section_header(badge_text: str, badge_class: str, title: str, caption: str):
    """Renders the colored badge + title + caption inside a section-card div is opened separately."""
    st.markdown(
        f"""
        <span class="section-badge {badge_class}">{badge_text}</span>
        <div class="section-title">{title}</div>
        <div class="section-caption">{caption}</div>
        """,
        unsafe_allow_html=True,
    )