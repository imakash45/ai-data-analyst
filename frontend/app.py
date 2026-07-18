import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import api_client as api
from styles import inject_custom_css, section_header

st.set_page_config(page_title="AI Data Analyst", layout="wide")
inject_custom_css()

if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "upload_info" not in st.session_state:
    st.session_state.upload_info = None
if "eda_result" not in st.session_state:
    st.session_state.eda_result = None
if "train_result" not in st.session_state:
    st.session_state.train_result = None
if "explain_result" not in st.session_state:
    st.session_state.explain_result = None
if "report_bytes" not in st.session_state:
    st.session_state.report_bytes = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.markdown(
    '<div class="hero-title"><span style="-webkit-text-fill-color: initial;">📊</span> AI Data Analyst</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hero-caption">Upload any CSV or Excel dataset to get auto-EDA, auto-ML, and an AI analyst chat layer.</div>',
    unsafe_allow_html=True,
)

if not api.check_backend_alive():
    st.error(
        f"Can't reach the backend at {api.BACKEND_URL}. "
        "It may be waking up from sleep (free-tier cold start can take up to a minute) — please refresh in a moment."
    )
    st.stop()


# ---------------- STEP 1: Upload ----------------
with st.container(border=True):
    section_header("STEP 1", "badge-upload", "Upload your dataset", "Start by uploading a CSV or Excel file to profile.")

    uploaded_file = st.file_uploader("CSV or Excel file", type=["csv", "xlsx", "xls"])

    if uploaded_file is not None and st.session_state.upload_info is None:
        with st.spinner("Uploading and profiling dataset..."):
            try:
                result = api.upload_file(uploaded_file.getvalue(), uploaded_file.name)
                st.session_state.session_id = result["session_id"]
                st.session_state.upload_info = result
            except api.APIError as e:
                st.error(f"Upload failed: {e}")

    if st.session_state.upload_info is not None:
        if st.button("Upload a different file"):
            st.session_state.session_id = None
            st.session_state.upload_info = None
            st.session_state.eda_result = None
            st.session_state.train_result = None
            st.session_state.explain_result = None
            st.session_state.report_bytes = None
            st.session_state.chat_history = []
            st.rerun()

    if st.session_state.upload_info is not None:
        info = st.session_state.upload_info
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", info["n_rows"])
        col2.metric("Columns", info["n_columns"])
        col3.metric("Duplicate rows", info["duplicate_row_count"])

        if info["warnings"]:
            for w in info["warnings"]:
                st.warning(w)

        st.subheader("Column overview")
        col_df = pd.DataFrame(
            [
                {
                    "Column": c["name"],
                    "Type": c["inferred_type"],
                    "Missing": f'{c["missing_count"]} ({c["missing_pct"]}%)',
                    "Unique values": c["unique_count"],
                    "Sample": ", ".join(str(v) for v in c["sample_values"][:3]),
                }
                for c in info["columns"]
            ]
        )
        st.dataframe(col_df, width='stretch', hide_index=True)


# ---------------- STEP 2: Clean ----------------
if st.session_state.upload_info is not None:
    with st.container(border=True):
        section_header("STEP 2", "badge-clean", "Clean your data", "Missing values are detected automatically, but you choose what to do with each column.")

        st.subheader("Column type overrides (optional)")
        st.caption("Force a column's inferred type before cleaning/training — e.g. reclassify a wrongly-dropped text column as categorical.")

        type_options = ["(no change)", "numeric", "categorical", "boolean", "datetime", "text"]
        dtype_overrides = {}

        override_cols = st.multiselect(
            "Select column(s) to override",
            options=[c["name"] for c in st.session_state.upload_info["columns"]],
            key="override_col_select",
        )

        for col_name in override_cols:
            col_info = next(c for c in st.session_state.upload_info["columns"] if c["name"] == col_name)
            chosen = st.selectbox(
                f'{col_name} — currently inferred as "{col_info["inferred_type"]}"',
                options=type_options,
                key=f"override_type_{col_name}",
            )
            if chosen != "(no change)":
                dtype_overrides[col_name] = chosen

        columns_with_missing = [c for c in st.session_state.upload_info["columns"] if c["missing_count"] > 0]

        rules = []
        if columns_with_missing:
            strategy_labels = {
                "leave_as_is": "Leave as-is",
                "mean": "Fill with mean",
                "median": "Fill with median",
                "mode": "Fill with mode",
                "drop_rows": "Drop rows",
            }
            for c in columns_with_missing:
                if c["inferred_type"] == "numeric":
                    options = ["leave_as_is", "mean", "median", "drop_rows"]
                else:
                    options = ["leave_as_is", "mode", "drop_rows"]

                choice = st.selectbox(
                    f'{c["name"]} — {c["missing_count"]} missing ({c["missing_pct"]}%)',
                    options=options,
                    format_func=lambda x: strategy_labels[x],
                    key=f"impute_{c['name']}",
                )
                rules.append({"column": c["name"], "imputation": choice})
        else:
            st.info("No missing values detected — nothing to configure here.")

        drop_dupes = st.checkbox(
            f'Drop {st.session_state.upload_info["duplicate_row_count"]} duplicate row(s)',
            value=st.session_state.upload_info["duplicate_row_count"] > 0,
            disabled=st.session_state.upload_info["duplicate_row_count"] == 0,
        )

        if st.button("Apply cleaning", type="primary"):
            with st.spinner("Applying cleaning rules..."):
                try:
                    result = api.clean_data(st.session_state.session_id, rules, drop_dupes, dtype_overrides)
                    st.success(
                        f'Done. Rows: {result["n_rows_before"]} -> {result["n_rows_after"]} '
                        f'({result["rows_dropped"]} dropped).'
                    )
                    if result["outliers"]:
                        with st.expander(f'{len(result["outliers"])} column(s) have outliers (IQR method)'):
                            for o in result["outliers"]:
                                st.write(
                                    f'**{o["column"]}**: {o["outlier_count"]} outlier(s), '
                                    f'expected range [{o["lower_bound"]}, {o["upper_bound"]}]'
                                )
                    st.session_state.upload_info["columns"] = result["columns"]
                    st.session_state.eda_result = None
                except api.APIError as e:
                    st.error(f"Cleaning failed: {e}")


# ---------------- STEP 3: EDA ----------------
if st.session_state.upload_info is not None:
    with st.container(border=True):
        section_header("STEP 3", "badge-eda", "Explore your data", "Auto-generated statistics, charts, and an AI-written insight paragraph.")

        if st.button("Run EDA", type="primary"):
            with st.spinner("Computing stats and generating insight..."):
                try:
                    st.session_state.eda_result = api.run_eda(st.session_state.session_id)
                except api.APIError as e:
                    st.error(f"EDA failed: {e}")

        eda = st.session_state.eda_result
        if eda is not None:
            st.subheader("🤖 AI-generated insight")
            if eda["insight_generated"]:
                st.info(eda["insight_text"])
            else:
                st.warning(eda["insight_text"])

            if eda["numeric_summary"]:
                st.subheader("Numeric columns")
                st.dataframe(pd.DataFrame(eda["numeric_summary"]), width='stretch', hide_index=True)

            if eda["categorical_summary"]:
                st.subheader("Categorical columns")
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "Column": c["column"],
                                "Top value": c["top_value"],
                                "Top value count": c["top_value_count"],
                                "Unique values": c["n_unique"],
                            }
                            for c in eda["categorical_summary"]
                        ]
                    ),
                    width='stretch',
                    hide_index=True,
                )

            if eda["missing_summary"]:
                st.subheader("Missing values")
                st.dataframe(pd.DataFrame(eda["missing_summary"]), width='stretch', hide_index=True)

            if eda["correlation"] is not None:
                st.subheader("Correlation matrix")
                corr = eda["correlation"]
                fig = go.Figure(
                    data=go.Heatmap(
                        z=corr["matrix"],
                        x=corr["columns"],
                        y=corr["columns"],
                        colorscale="RdBu",
                        zmin=-1,
                        zmax=1,
                        text=[[f"{v:.2f}" for v in row] for row in corr["matrix"]],
                        texttemplate="%{text}",
                    )
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, width='stretch')

            if eda["distributions"]:
                st.subheader("Distributions")
                dist_cols = st.columns(2)
                for i, dist in enumerate(eda["distributions"]):
                    bin_labels = [f'{b["bin_start"]:.1f}-{b["bin_end"]:.1f}' for b in dist["bins"]]
                    counts = [b["count"] for b in dist["bins"]]
                    fig = px.bar(x=bin_labels, y=counts, title=dist["column"], labels={"x": "", "y": "Count"})
                    fig.update_layout(height=300)
                    dist_cols[i % 2].plotly_chart(fig, width='stretch')

            categorical_value_counts = [c for c in eda["categorical_summary"] if c["value_counts"]]
            if categorical_value_counts:
                st.subheader("Category breakdowns")
                cat_cols = st.columns(2)
                for i, c in enumerate(categorical_value_counts):
                    fig = px.bar(
                        x=list(c["value_counts"].keys()),
                        y=list(c["value_counts"].values()),
                        title=c["column"],
                        labels={"x": "", "y": "Count"},
                    )
                    fig.update_layout(height=300)
                    cat_cols[i % 2].plotly_chart(fig, width='stretch')


# ---------------- STEP 4: Train ----------------
if st.session_state.upload_info is not None:
    with st.container(border=True):
        section_header("STEP 4", "badge-train", "Train models", "Pick a target column. Task type (regression/classification) is auto-detected.")

        column_names = [c["name"] for c in st.session_state.upload_info["columns"]]
        target_column = st.selectbox("Target column", options=column_names, key="target_column_select")

        excluded_columns = st.multiselect(
            "Exclude column(s) from training (optional)",
            options=[c for c in column_names if c != target_column],
            key="excluded_columns_select",
            help="Force these columns out of training entirely — useful for irrelevant datetime columns, IDs, or anything you know shouldn't be a feature.",
        )

        if st.button("Train models", type="primary"):
            with st.spinner("Training Linear/Logistic Regression, Random Forest, and XGBoost..."):
                try:
                    st.session_state.train_result = api.train_models(
                        st.session_state.session_id, target_column, excluded_columns
                    )
                except api.APIError as e:
                    st.error(f"Training failed: {e}")

        train_result = st.session_state.get("train_result")
        if train_result is not None:
            st.subheader("Training summary")
            col1, col2, col3 = st.columns(3)
            col1.metric("Task type", train_result["task_type"].capitalize())
            col2.metric("Rows used", train_result["n_rows_used"])
            col3.metric("Features used", train_result["n_features_used"])

            if train_result["dropped_columns"]:
                with st.expander(f'{len(train_result["dropped_columns"])} column(s) dropped before training'):
                    for d in train_result["dropped_columns"]:
                        st.write(f'**{d["column"]}**: {d["reason"]}')

            st.subheader("Model comparison")
            metric_keys = [k for k, v in train_result["results"][0]["metrics"].items() if v is not None]

            table_rows = []
            for r in train_result["results"]:
                row = {"Model": r["model_name"]}
                for k in metric_keys:
                    row[k.upper()] = r["metrics"][k]
                row["Best"] = "✅" if r["is_best"] else ""
                table_rows.append(row)

            st.dataframe(pd.DataFrame(table_rows), width='stretch', hide_index=True)
            st.success(f'Best model: **{train_result["best_model"]}**')

# ---------------- STEP 5: Explain ----------------
if st.session_state.get("train_result") is not None:
    with st.container(border=True):
        section_header("STEP 5", "badge-explain", "Explain the model", "SHAP feature importance for the best model selected above.")

        if st.button("Explain model", type="primary"):
            with st.spinner("Computing SHAP values..."):
                try:
                    st.session_state.explain_result = api.explain_model(st.session_state.session_id)
                except api.APIError as e:
                    st.error(f"Explain failed: {e}")

        explain_result = st.session_state.get("explain_result")
        if explain_result is not None:
            st.caption(
                f'Model: **{explain_result["model_name"]}** · '
                f'Rows sampled for SHAP: {explain_result["n_rows_sampled"]}'
            )

            importances = explain_result["feature_importances"]
            fig = px.bar(
                x=[f["importance"] for f in importances],
                y=[f["feature"] for f in importances],
                orientation="h",
                labels={"x": "Mean |SHAP value|", "y": "Feature"},
                title="Feature importance",
            )
            fig.update_layout(height=max(300, len(importances) * 30), yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, width='stretch')


# ---------------- STEP 7: Report ----------------
if st.session_state.upload_info is not None:
    with st.container(border=True):
        section_header("STEP 6", "badge-report", "Download report", "A PDF combining dataset overview, AI insight, charts, and model results (if trained).")

        if st.button("Generate report", type="primary"):
            with st.spinner("Building PDF report..."):
                try:
                    st.session_state.report_bytes = api.generate_report(st.session_state.session_id)
                except api.APIError as e:
                    st.error(f"Report generation failed: {e}")

        report_bytes = st.session_state.get("report_bytes")
        if report_bytes is not None:
            st.download_button(
                label="Download PDF report",
                data=report_bytes,
                file_name=f"report_{st.session_state.session_id[:8]}.pdf",
                mime="application/pdf",
            )


# ---------------- STEP 6: Chat (input pinned to bottom via st.chat_input) ----------------
if st.session_state.upload_info is not None:
    with st.container(border=True):
        section_header("STEP 7", "badge-chat", "Ask the AI analyst", "Answers are grounded in the stats already computed above (EDA, and /train results if run).")

        for entry in st.session_state.chat_history:
            st.chat_message("user").write(entry["question"])
            st.chat_message("assistant").write(entry["answer"])

    question = st.chat_input("Ask a question about your dataset...")

    if question:
        with st.spinner("Thinking..."):
            try:
                result = api.chat_with_dataset(st.session_state.session_id, question)
                st.session_state.chat_history.append(
                    {"question": question, "answer": result["answer"]}
                )
                st.rerun()
            except api.APIError as e:
                st.error(f"Chat failed: {e}")