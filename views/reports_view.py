"""
Export Reports View for DataSense AI.
Provides one-click downloads for PDF executive reports, multi-tab Excel workbooks,
and cleaned CSV data files.
"""

from datetime import datetime
import streamlit as st
import pandas as pd
from utils.ui_components import render_page_header, render_section_header, render_kpi_card
from backend.report_generator import ReportGenerator


def render_reports_view():
    df = st.session_state.get("df_cleaned")
    if df is None:
        st.warning("⚠️ No dataset loaded. Please upload a dataset in 'Upload Dataset' first.")
        return

    dataset_name = st.session_state.get("dataset_name", "Business_Dataset")
    profile_data = st.session_state.get("profile_data", {})
    kpis = st.session_state.get("kpis", [])
    insights = st.session_state.get("insights", [])
    cleaning_engine = st.session_state.get("cleaning_engine")
    cleaning_log = cleaning_engine.get_log() if cleaning_engine else []
    eda_summary = st.session_state.get("eda_summary")

    clean_base_name = dataset_name.rsplit(".", 1)[0].replace(" ", "_")
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M")

    render_page_header("Export Analytical Reports", f"Generate downloadable deliverables for: {dataset_name}", icon="📑", eyebrow="Report Export")

    c1, c2, c3 = st.columns(3, gap="large")

    # 1. PDF Executive Report
    with c1:
        st.markdown("### 📄 Executive PDF Report")
        st.markdown(
            """
            Comprehensive management-ready PDF report containing:
            - Dataset Overview & Health Scorecard
            - Key Performance Indicators (KPIs)
            - Strategic Business Findings & Recommendations
            - Full Data Cleaning Audit Trail
            - Analytical Assumptions & Disclaimers
            """
        )
        with st.spinner("Compiling PDF report..."):
            pdf_bytes = ReportGenerator.generate_pdf_report(
                df=df,
                profile_data=profile_data,
                kpis=kpis,
                insights=insights,
                cleaning_log=cleaning_log,
                dataset_name=dataset_name
            )

        st.download_button(
            label="⬇️ Download Executive PDF Report",
            data=pdf_bytes,
            file_name=f"DataSense_Report_{clean_base_name}_{timestamp_str}.pdf",
            mime="application/pdf",
            type="primary", width='stretch'
        )

    # 2. Multi-Tab Excel Workbook
    with c2:
        st.markdown("### 📊 Multi-Tab Excel Workbook")
        st.markdown(
            """
            Multi-sheet analytical workbook including:
            - **Cleaned_Data**: Fully remediated dataset
            - **Executive_KPIs**: Top calculated business metrics
            - **Column_Quality_Profile**: Feature-by-feature audit
            - **Descriptive_Statistics**: Mean, std, quartiles, skew
            - **Cleaning_Audit_Log**: Operation history
            """
        )
        with st.spinner("Generating Excel workbook..."):
            excel_bytes = ReportGenerator.generate_excel_workbook(
                cleaned_df=df,
                profile_data=profile_data,
                kpis=kpis,
                cleaning_log=cleaning_log,
                eda_summary=eda_summary
            )

        st.download_button(
            label="⬇️ Download Excel Workbook (.xlsx)",
            data=excel_bytes,
            file_name=f"DataSense_Workbook_{clean_base_name}_{timestamp_str}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary", width='stretch'
        )

    # 3. Cleaned CSV Dataset
    with c3:
        st.markdown("### 💾 Cleaned Dataset (CSV)")
        st.markdown(
            """
            Direct export of the transformed and remediated data:
            - All missing values handled
            - Outliers clamped or resolved
            - Inconsistent text and casing normalized
            - Currency and formatted strings converted to numbers
            - Ready for downstream ML pipelines
            """
        )
        csv_data = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇️ Download Cleaned CSV",
            data=csv_data,
            file_name=f"DataSense_Cleaned_{clean_base_name}_{timestamp_str}.csv",
            mime="text/csv",
            type="primary", width='stretch'
        )

    st.markdown("---")
    st.markdown("### 📋 Live Data Quality & Profiling Summary")
    col_details = profile_data.get("column_details", [])
    if col_details:
        st.dataframe(pd.DataFrame(col_details), width='stretch', hide_index=True)

