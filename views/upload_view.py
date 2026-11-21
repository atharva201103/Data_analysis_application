"""
Upload Dataset View for DataSense AI.
Supports CSV, XLSX, XLS uploads, multi-sheet selection, and quick sample dataset loading.
"""

import os
import streamlit as st
import pandas as pd
from backend.data_loader import DataLoader
from backend.column_detector import ColumnDetector
from backend.cleaning_engine import CleaningEngine
from backend.data_profiler import DataProfiler
from backend.kpi_engine import KPIEngine
from backend.insight_engine import InsightEngine
from utils.ui_components import render_page_header, render_section_header, render_kpi_card
from utils.helpers import format_bytes


def render_upload_view():
    render_page_header("Upload Dataset", "Import your tabular business data in CSV, XLSX, or XLS format", icon="📁", eyebrow="Data Import")

    col_upload, col_sample = st.columns([1.6, 1.0], gap="large")

    with col_upload:
        st.markdown("### 📤 Upload Your Business Dataset")
        uploaded_file = st.file_uploader(
            "Drag and drop file here or click to browse",
            type=["csv", "xlsx", "xls"],
            help="Maximum supported file size is 50 MB."
        )

        if uploaded_file is not None:
            filename = uploaded_file.name
            file_bytes = uploaded_file.getvalue()
            file_size_formatted = format_bytes(len(file_bytes))

            # If Excel, inspect sheets
            sheet_selected = None
            if filename.lower().endswith((".xlsx", ".xls")):
                sheets = DataLoader.get_excel_sheet_names(file_bytes)
                if len(sheets) > 1:
                    sheet_selected = st.selectbox(
                        "📑 Select Worksheet",
                        options=sheets,
                        help="This Excel file contains multiple worksheets. Choose which one to analyze."
                    )
                elif len(sheets) == 1:
                    sheet_selected = sheets[0]

            if st.button("🚀 Process & Load Dataset", type="primary", width='stretch'):
                with st.spinner("Analyzing and parsing dataset..."):
                    df, err = DataLoader.load_dataset(file_bytes, filename, sheet_name=sheet_selected)
                    if err:
                        st.error(f"Error loading file: {err}")
                    else:
                        _initialize_dataset(df, filename)
                        st.success(f"Successfully loaded '{filename}' ({len(df):,} rows, {len(df.columns)} columns)!")
                        st.rerun()

    with col_sample:
        st.markdown("### 🧪 Or Try a Pre-Loaded Sample")
        st.markdown("Explore DataSense AI immediately with pre-configured business datasets containing realistic anomalies:")

        sample_choice = st.selectbox(
            "Select Business Domain Sample",
            options=[
                "Retail Sales (Sales, Profit, Quantities, Margin Issues)",
                "HR Analytics (Workforce, Salaries, Attrition, Overtime)",
                "E-Commerce Orders (Orders, Shipping, Discounting, Categories)"
            ]
        )

        sample_paths = {
            "Retail Sales (Sales, Profit, Quantities, Margin Issues)": "sample_datasets/retail_sales_sample.csv",
            "HR Analytics (Workforce, Salaries, Attrition, Overtime)": "sample_datasets/hr_analytics_sample.csv",
            "E-Commerce Orders (Orders, Shipping, Discounting, Categories)": "sample_datasets/ecommerce_orders_sample.csv"
        }

        if st.button("⚡ Load Selected Sample", width='stretch'):
            target_file = sample_paths[sample_choice]
            with st.spinner("Loading sample dataset..."):
                df, err = DataLoader.load_dataset(target_file, os.path.basename(target_file))
                if err:
                    st.error(f"Failed to load sample: {err}")
                else:
                    _initialize_dataset(df, os.path.basename(target_file))
                    st.success(f"Loaded {os.path.basename(target_file)}!")
                    st.rerun()

    # If dataset is currently active, show preview
    if st.session_state.get("df_cleaned") is not None:
        df = st.session_state["df_cleaned"]
        dataset_name = st.session_state.get("dataset_name", "Active Dataset")

        st.markdown("---")
        st.markdown(f"### 📋 Current Active Dataset: `{dataset_name}`")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            render_kpi_card("Total Rows", f"{len(df):,}", icon="database")
        with c2:
            render_kpi_card("Total Columns", str(len(df.columns)), icon="columns")
        with c3:
            render_kpi_card("Missing Cells", f"{df.isna().sum().sum():,}", icon="alert-triangle")
        with c4:
            render_kpi_card("Memory Size", format_bytes(int(df.memory_usage(deep=True).sum())), icon="hard-drive")

        st.markdown("#### 🔍 First 10 Records Preview")
        st.dataframe(df.head(10), width='stretch')

        st.markdown("#### 🏷️ Inferred Column Data Types & Roles")
        roles = st.session_state.get("column_roles", {})
        type_summary = []
        for col in df.columns:
            meta = roles.get(col, {})
            type_summary.append({
                "Column Name": col,
                "Pandas Dtype": str(df[col].dtype),
                "Inferred Role": meta.get("role", "General"),
                "Confidence": f"{meta.get('confidence', 0.5)*100:.0f}%",
                "Sample Value": str(df[col].dropna().iloc[0]) if len(df[col].dropna()) > 0 else "N/A"
            })
        st.dataframe(pd.DataFrame(type_summary), width='stretch', hide_index=True)

        # Clear dataset option
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑️ Clear Current Dataset from Session", type="secondary"):
            _clear_session_data()
            st.rerun()


def _initialize_dataset(df: pd.DataFrame, filename: str):
    """Store dataset and run foundational profiling."""
    st.session_state["df_original"] = df.copy(deep=True)
    st.session_state["df_cleaned"] = df.copy(deep=True)
    st.session_state["dataset_name"] = filename

    # Column roles
    detector = ColumnDetector(overrides=st.session_state.get("column_role_overrides", {}))
    column_roles = detector.detect_all(df)
    st.session_state["column_roles"] = column_roles

    # Cleaning Engine
    st.session_state["cleaning_engine"] = CleaningEngine(df)

    # Profiling Data
    profile_data = DataProfiler.profile_dataset(df, column_roles=column_roles)
    st.session_state["profile_data"] = profile_data

    # KPIs
    domain = KPIEngine.identify_domain(df, column_roles)
    st.session_state["domain"] = domain
    kpis = KPIEngine.compute_kpis(df, column_roles, domain=domain)
    st.session_state["kpis"] = kpis

    # Insights
    insight_engine = InsightEngine()
    insights = insight_engine.generate_all_insights(df, column_roles, domain=domain)
    st.session_state["insights"] = insights

    # Chat history reset
    st.session_state["chat_messages"] = []


def _clear_session_data():
    """Wipe current dataset from session."""
    for key in ["df_original", "df_cleaned", "dataset_name", "column_roles", "cleaning_engine", "profile_data", "kpis", "insights", "chat_messages"]:
        if key in st.session_state:
            del st.session_state[key]

