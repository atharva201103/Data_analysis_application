"""
Exploratory Data Analysis (EDA) View for DataSense AI.
Provides descriptive statistics, categorical frequency distributions, correlation matrices,
and interactive feature distribution exploration.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from utils.ui_components import render_page_header, render_section_header
from backend.eda_engine import EDAEngine
from backend.visualization_engine import PLOTLY_LAYOUT_DEFAULTS


def render_eda_view():
    df = st.session_state.get("df_cleaned")
    if df is None:
        st.warning("⚠️ No dataset loaded. Please upload a dataset in 'Upload Dataset' first.")
        return

    dataset_name = st.session_state.get("dataset_name", "Dataset")
    render_page_header("Exploratory Data Analysis", f"Statistical audit for {dataset_name} · {len(df):,} records", icon="📈", eyebrow="EDA Module")

    tab_desc, tab_cat, tab_corr, tab_feature = st.tabs([
        "1. Descriptive Statistics",
        "2. Categorical Distributions",
        "3. Correlation & Covariance",
        "4. Feature Inspector"
    ])

    # -------------------------------------------------------------
    # TAB 1: DESCRIPTIVE STATISTICS
    # -------------------------------------------------------------
    with tab_desc:
        st.markdown("### 📊 Comprehensive Numerical Summaries")
        num_summary = EDAEngine.get_numerical_summary(df)
        if not num_summary.empty:
            st.dataframe(num_summary, width='stretch', hide_index=True)
            st.session_state["eda_summary"] = num_summary
            st.caption("Includes central tendency (Mean, Median, Mode), dispersion (Std, Variance, IQR), and distribution shape (Skewness, Kurtosis).")
        else:
            st.info("No numerical columns found in the dataset to compute descriptive statistics.")

    # -------------------------------------------------------------
    # TAB 2: CATEGORICAL DISTRIBUTIONS
    # -------------------------------------------------------------
    with tab_cat:
        st.markdown("### 🏷️ Categorical Value Frequencies")
        cat_summary = EDAEngine.get_categorical_summary(df)
        if cat_summary:
            sel_cat = st.selectbox("Select Categorical Feature to Inspect", options=list(cat_summary.keys()))
            meta = cat_summary[sel_cat]

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Unique Classes", meta["unique_count"])
            with c2:
                st.metric("Dominant Class", str(meta["most_frequent"]), f"{meta['most_frequent_count']:,} rows")
            with c3:
                st.metric("Least Frequent Class", str(meta["least_frequent"]), f"{meta['least_frequent_count']:,} rows")

            # Chart of top categories
            top_cats_df = pd.DataFrame(list(meta["top_categories"].items()), columns=["Category", "Frequency"])
            fig = px.bar(
                top_cats_df,
                x="Category",
                y="Frequency",
                color="Frequency",
                color_continuous_scale="Purples",
                title=f"Class Frequency: {sel_cat} (Top 10)"
            )
            fig.update_layout(**PLOTLY_LAYOUT_DEFAULTS)
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("No categorical features found in this dataset.")

    # -------------------------------------------------------------
    # TAB 3: CORRELATION & COVARIANCE
    # -------------------------------------------------------------
    with tab_corr:
        st.markdown("### 🔥 Multi-Feature Correlation Matrix")
        corr_method = st.radio("Correlation Metric", options=["pearson", "spearman"], horizontal=True)
        corr_df, pairs = EDAEngine.get_correlation_matrix(df, method=corr_method)

        if not corr_df.empty:
            fig = px.imshow(
                corr_df,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                title=f"{corr_method.capitalize()} Correlation Heatmap"
            )
            fig.update_layout(**PLOTLY_LAYOUT_DEFAULTS)
            st.plotly_chart(fig, width='stretch')

            if pairs:
                st.markdown("#### 🎯 Strongest Pairwise Correlations")
                st.dataframe(pd.DataFrame(pairs[:10]), width='stretch', hide_index=True)
        else:
            st.info("At least two numerical columns are required to construct a correlation matrix.")

    # -------------------------------------------------------------
    # TAB 4: FEATURE INSPECTOR
    # -------------------------------------------------------------
    with tab_feature:
        st.markdown("### 🔬 Single Feature Distribution Inspector")
        inspect_col = st.selectbox("Choose Column to Inspect", options=df.columns.tolist())

        if pd.api.types.is_numeric_dtype(df[inspect_col]):
            s = df[inspect_col].dropna()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Mean", f"{s.mean():,.2f}")
            c2.metric("Median", f"{s.median():,.2f}")
            c3.metric("Std Dev", f"{s.std():,.2f}")
            c4.metric("IQR", f"{s.quantile(0.75) - s.quantile(0.25):,.2f}")

            fig = px.histogram(
                df,
                x=inspect_col,
                marginal="box",
                nbins=35,
                title=f"Distribution & Outlier Boxplot: {inspect_col}",
                color_discrete_sequence=["#06B6D4"]
            )
            fig.update_layout(**PLOTLY_LAYOUT_DEFAULTS)
            st.plotly_chart(fig, width='stretch')
        else:
            vc = df[inspect_col].value_counts().head(15).reset_index()
            vc.columns = [inspect_col, "Count"]
            fig = px.bar(
                vc,
                x=inspect_col,
                y="Count",
                title=f"Distribution of {inspect_col}",
                color="Count",
                color_continuous_scale="Teal"
            )
            fig.update_layout(**PLOTLY_LAYOUT_DEFAULTS)
            st.plotly_chart(fig, width='stretch')

