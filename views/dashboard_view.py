"""
Executive Dashboard View for DataSense AI.
Provides high-level KPIs, health score, top visual trends, and core business findings.
"""

import streamlit as st
import pandas as pd
from utils.ui_components import (
    render_page_header, render_section_header, render_kpi_card, render_health_score_card, render_insight_card
)
from backend.visualization_engine import VisualizationEngine


def render_dashboard_view():
    df = st.session_state.get("df_cleaned")

    if df is None:
        _render_empty_state()
        return

    dataset_name = st.session_state.get("dataset_name", "Business Dataset")
    domain = st.session_state.get("domain", "general").replace("_", " ").title()

    render_page_header("Executive Analytics Dashboard", f"Active Dataset: {dataset_name}  •  Domain: {domain}  •  {len(df):,} records", icon="📊", eyebrow="Analytics Dashboard")

    # 1. Top Business KPIs
    kpis = st.session_state.get("kpis", [])
    if kpis:
        st.markdown("### 🎯 Executive Performance Indicators")
        cols = st.columns(min(4, len(kpis)))
        for i, kpi in enumerate(kpis[:4]):
            with cols[i % len(cols)]:
                render_kpi_card(
                    title=kpi.get("title", "KPI"),
                    value=str(kpi.get("value", "N/A")),
                    subtext=kpi.get("subtext"),
                    delta=kpi.get("delta"),
                    delta_positive=kpi.get("delta_positive", True),
                    icon=kpi.get("icon", "📊")
                )
        st.markdown("<br>", unsafe_allow_html=True)

    # 2. Split: Health Score (Left) + Top Insights (Right)
    col_health, col_insights = st.columns([1.1, 1.9], gap="large")

    with col_health:
        st.markdown("### 🩺 Dataset Health")
        profile = st.session_state.get("profile_data", {})
        score = profile.get("health_score", 100)
        grade = profile.get("health_grade", "A")
        explanation = profile.get("health_explanation", "Health score represents complete data integrity.")

        render_health_score_card(score, grade, explanation)

        # Health Breakdown Table
        breakdown = profile.get("score_breakdown", {})
        if breakdown:
            st.markdown("<br>", unsafe_allow_html=True)
            b_df = pd.DataFrame([
                {"Dimension": "Cell Completeness", "Earned": f"{breakdown.get('completeness_score', 0)} / {breakdown.get('completeness_max', 35)} pts"},
                {"Dimension": "Row Uniqueness", "Earned": f"{breakdown.get('uniqueness_score', 0)} / {breakdown.get('uniqueness_max', 25)} pts"},
                {"Dimension": "Feature Richness", "Earned": f"{breakdown.get('richness_score', 0)} / {breakdown.get('richness_max', 20)} pts"},
                {"Dimension": "Column Integrity", "Earned": f"{breakdown.get('column_quality_score', 0)} / {breakdown.get('column_quality_max', 20)} pts"}
            ])
            st.dataframe(b_df, width='stretch', hide_index=True)

    with col_insights:
        st.markdown("### 💡 Strategic Business Highlights")
        insights = st.session_state.get("insights", [])
        if insights:
            for ins in insights[:3]:
                render_insight_card(
                    title=ins.get("title", "Insight"),
                    observation=ins.get("observation", ""),
                    implication=ins.get("implication", ""),
                    recommendation=ins.get("recommendation", ""),
                    metrics_summary=ins.get("evidence"),
                    category=ins.get("category", "general")
                )
        else:
            st.info("No specific business rule triggers detected for this dataset structure. Explore EDA for descriptive findings.")

    # 3. Recommended Automated Charts
    st.markdown("---")
    st.markdown("### 📈 Core Business Visualizations")
    column_roles = st.session_state.get("column_roles", {})
    charts = VisualizationEngine.generate_recommended_charts(df, column_roles, max_charts=4)

    if charts:
        for i in range(0, len(charts), 2):
            row_charts = charts[i:i+2]
            c_cols = st.columns(len(row_charts))
            for j, ch in enumerate(row_charts):
                with c_cols[j]:
                    st.plotly_chart(ch["figure"], width='stretch')
                    st.caption(ch.get("description", ""))
    else:
        st.info("Generate custom visual charts in the Data Visualization module.")


def _render_empty_state():
    render_page_header("Welcome to DataSense AI", "Turn Raw Data Into Business Intelligence. Upload a dataset to begin.", icon="👋", eyebrow="DataSense AI")
    st.info("No dataset is currently loaded in your session. Get started by uploading a dataset or loading a sample.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📁 Go to Dataset Upload", type="primary", width='stretch'):
            st.session_state["nav_selection"] = "Upload Dataset"
            st.rerun()
    with col2:
        if st.button("⚡ Load Quick Retail Sales Sample", width='stretch'):
            from backend.data_loader import DataLoader
            df, _ = DataLoader.load_dataset("sample_datasets/retail_sales_sample.csv", "retail_sales_sample.csv")
            from views.upload_view import _initialize_dataset
            _initialize_dataset(df, "retail_sales_sample.csv")
            st.rerun()

