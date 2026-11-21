"""
Data Visualization View for DataSense AI.
Features automated domain-aware charts and an interactive Custom Chart Studio.
"""

import streamlit as st
import pandas as pd
from utils.ui_components import render_page_header, render_section_header
from backend.visualization_engine import VisualizationEngine


def render_visualization_view():
    df = st.session_state.get("df_cleaned")
    if df is None:
        st.warning("⚠️ No dataset loaded. Please upload a dataset in 'Upload Dataset' first.")
        return

    dataset_name = st.session_state.get("dataset_name", "Dataset")
    column_roles = st.session_state.get("column_roles", {})

    render_page_header("Data Visualization Studio", f"Interactive visual analytics for: {dataset_name}", icon="📉", eyebrow="Visualization")

    tab_auto, tab_custom = st.tabs(["1. Automated Domain Visualizations", "2. Custom Chart Studio"])

    # -------------------------------------------------------------
    # TAB 1: AUTOMATED VISUALIZATIONS
    # -------------------------------------------------------------
    with tab_auto:
        st.markdown("### 🤖 Recommended Domain Charts")
        st.caption("Intelligently generated based on detected column roles (Date, Revenue, Profit, Category, Region).")

        charts = VisualizationEngine.generate_recommended_charts(df, column_roles, max_charts=6)
        if charts:
            for i in range(0, len(charts), 2):
                row_charts = charts[i:i+2]
                cols = st.columns(len(row_charts))
                for j, ch in enumerate(row_charts):
                    with cols[j]:
                        st.plotly_chart(ch["figure"], width='stretch')
                        st.caption(f"**{ch['title']}**: {ch.get('description', '')}")
                        st.markdown("<br>", unsafe_allow_html=True)
        else:
            st.info("No auto-recommended charts available for this dataset structure. Build custom charts in the Studio tab!")

    # -------------------------------------------------------------
    # TAB 2: CUSTOM CHART STUDIO
    # -------------------------------------------------------------
    with tab_custom:
        st.markdown("### 🎨 Custom Interactive Chart Studio")
        st.markdown("Build customized Plotly charts by selecting axes, dimensions, and aggregation functions:")

        c_t1, c_t2, c_t3, c_t4 = st.columns(4)

        with c_t1:
            chart_type = st.selectbox(
                "Chart Type",
                options=["Bar", "Horizontal Bar", "Line", "Area", "Scatter", "Pie / Donut", "Histogram", "Box Plot", "Treemap"]
            )

        with c_t2:
            x_col = st.selectbox("X-Axis / Primary Dimension", options=df.columns.tolist())

        with c_t3:
            num_cols = df.select_dtypes(include=["number"]).columns.tolist()
            y_options = ["None"] + num_cols if chart_type in ["Histogram"] else (num_cols if num_cols else df.columns.tolist())
            y_col = st.selectbox("Y-Axis / Metric", options=y_options)
            if y_col == "None":
                y_col = None

        with c_t4:
            cat_options = ["None"] + df.columns.tolist()
            color_col = st.selectbox("Color / Grouping Segment", options=cat_options)
            if color_col == "None":
                color_col = None

        # Additional options for aggregations
        c_opt1, c_opt2 = st.columns(2)
        with c_opt1:
            agg_func = st.selectbox("Aggregation Method", options=["sum", "mean", "median", "count", "min", "max"])
        with c_opt2:
            barmode = st.selectbox("Bar Stacking Mode", options=["group", "stack", "relative"])

        try:
            fig = VisualizationEngine.create_custom_chart(
                df=df,
                chart_type=chart_type,
                x_col=x_col,
                y_col=y_col,
                color_col=color_col,
                agg_func=agg_func,
                barmode=barmode
            )
            st.plotly_chart(fig, width='stretch')
        except Exception as e:
            st.error(f"Could not render custom chart with chosen configuration: {str(e)}")

