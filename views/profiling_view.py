"""
Data Profiling View for DataSense AI.
Presents structural metrics, column attributes, and automated health score diagnostics.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from utils.ui_components import (
    render_page_header, render_section_header, render_kpi_card, render_health_score_card
)
from backend.visualization_engine import PLOTLY_LAYOUT_DEFAULTS


def render_profiling_view():
    df = st.session_state.get("df_cleaned")
    if df is None:
        st.warning("⚠️ No dataset loaded. Please upload a dataset in 'Upload Dataset' first.")
        return

    profile = st.session_state.get("profile_data", {})
    dataset_name = st.session_state.get("dataset_name", "Dataset")

    render_page_header("Automated Data Profiling", f"Deep structural audit for: {dataset_name}", icon="🔍", eyebrow="Data Profiling")

    # 1. High-Level Metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        render_kpi_card("Total Rows", f"{profile.get('total_rows', len(df)):,}", icon="database")
    with c2:
        render_kpi_card("Total Columns", str(profile.get("total_columns", len(df.columns))), icon="columns")
    with c3:
        render_kpi_card("Missing Cells", f"{profile.get('total_missing_cells', 0):,}", subtext=f"{profile.get('total_missing_percentage', 0):.1f}% of cells", icon="alert-triangle")
    with c4:
        render_kpi_card("Duplicate Rows", f"{profile.get('duplicate_rows', 0):,}", subtext=f"{profile.get('duplicate_percentage', 0):.1f}% duplicate", icon="copy")
    with c5:
        render_kpi_card("Memory Footprint", profile.get("memory_formatted", "N/A"), icon="hard-drive")

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Health Score Card & Formula Explanation
    col_score, col_expl = st.columns([1, 1.4], gap="large")
    with col_score:
        render_health_score_card(
            profile.get("health_score", 100),
            profile.get("health_grade", "A"),
            profile.get("health_explanation", "")
        )

    with col_expl:
        st.markdown("### 📐 Health Score Scoring Rubric")
        st.markdown(
            """
            The **Dataset Health Score** is calculated through an objective weighted rubric evaluating four quality dimensions:
            - **Completeness (35 pts max):** Deducts points proportionally for missing cells across the entire dataset.
            - **Row Uniqueness (25 pts max):** Penalizes duplicate rows that skew statistical aggregates.
            - **Feature Richness (20 pts max):** Penalizes constant (zero-variance) columns offering no predictive value.
            - **Column Quality (20 pts max):** Penalizes features with catastrophic missingness (>50% nulls).
            """
        )
        b = profile.get("score_breakdown", {})
        if b:
            score_df = pd.DataFrame([
                {"Dimension": "Cell Completeness", "Earned Score": b.get("completeness_score"), "Max Score": b.get("completeness_max")},
                {"Dimension": "Row Uniqueness", "Earned Score": b.get("uniqueness_score"), "Max Score": b.get("uniqueness_max")},
                {"Dimension": "Feature Richness", "Earned Score": b.get("richness_score"), "Max Score": b.get("richness_max")},
                {"Dimension": "Column Quality", "Earned Score": b.get("column_quality_score"), "Max Score": b.get("column_quality_max")}
            ])
            st.dataframe(score_df, width='stretch', hide_index=True)

    st.markdown("---")

    # 3. Column Quality Breakdown Table
    st.markdown("### 📋 Column-Level Quality Audit")
    details = profile.get("column_details", [])
    if details:
        det_df = pd.DataFrame(details)
        # Rename for clean presentation
        display_df = det_df.rename(columns={
            "column_name": "Column",
            "data_type": "Data Type",
            "general_type": "General Type",
            "inferred_role": "Inferred Role",
            "missing_count": "Null Count",
            "missing_percentage": "Null %",
            "unique_count": "Uniques",
            "min": "Min",
            "max": "Max",
            "mean": "Mean",
            "median": "Median",
            "is_constant": "Constant?",
            "is_identifier": "ID?"
        })
        st.dataframe(display_df, width='stretch', hide_index=True)

    # 4. Missing Value Visual Distribution
    if profile.get("total_missing_cells", 0) > 0:
        st.markdown("### ⚠️ Missing Values Distribution")
        missing_cols_data = [d for d in details if d["missing_count"] > 0]
        if missing_cols_data:
            m_df = pd.DataFrame(missing_cols_data).sort_values(by="missing_percentage", ascending=True)
            fig = px.bar(
                m_df,
                y="column_name",
                x="missing_percentage",
                orientation="h",
                color="missing_percentage",
                color_continuous_scale="Reds",
                title="Missing Value Percentage by Column (%)"
            )
            fig.update_layout(**PLOTLY_LAYOUT_DEFAULTS)
            fig.update_xaxes(title_text="Missing Percentage (%)")
            fig.update_yaxes(title_text="Column Name")
            st.plotly_chart(fig, width='stretch')

