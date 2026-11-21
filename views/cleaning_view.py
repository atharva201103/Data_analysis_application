"""
Data Cleaning View for DataSense AI.
Provides interactive, non-destructive data cleaning tools with before-and-after tracking,
preview mechanism, and full audit logging.
"""

import io
import streamlit as st
import pandas as pd
from utils.ui_components import render_page_header, render_section_header, render_kpi_card
from backend.cleaning_engine import CleaningEngine
from backend.data_profiler import DataProfiler
from backend.kpi_engine import KPIEngine
from backend.insight_engine import InsightEngine


def render_cleaning_view():
    df = st.session_state.get("df_cleaned")
    if df is None:
        st.warning("⚠️ No dataset loaded. Please upload a dataset in 'Upload Dataset' first.")
        return

    cleaning_engine: CleaningEngine = st.session_state.get("cleaning_engine")
    if cleaning_engine is None:
        cleaning_engine = CleaningEngine(df)
        st.session_state["cleaning_engine"] = cleaning_engine

    render_page_header("Interactive Data Cleaning", "Remediate missing values, duplicates, outliers, and type errors — non-destructively", icon="🧹", eyebrow="Data Cleaning")

    # 1. Before & After Summary Banner
    summary = cleaning_engine.get_before_after_summary()
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        render_kpi_card("Current Rows", f"{summary['cleaned_rows']:,}", subtext=f"Original: {summary['original_rows']:,}", icon="database")
    with c2:
        render_kpi_card("Current Columns", str(summary['cleaned_cols']), subtext=f"Original: {summary['original_cols']}", icon="columns")
    with c3:
        render_kpi_card("Missing Resolved", f"{summary['missing_cells_resolved']:,}", subtext=f"Remaining: {summary['cleaned_missing_cells']:,}", icon="check-circle")
    with c4:
        render_kpi_card("Duplicates Dropped", f"{summary['duplicates_removed']:,}", subtext=f"Remaining: {summary['cleaned_duplicates']:,}", icon="copy")
    with c5:
        render_kpi_card("Operations Applied", str(summary['total_operations_applied']), icon="zap")

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Automated Cleaning Recommendations
    recs = cleaning_engine.get_cleaning_recommendations()
    if recs:
        with st.expander(f"💡 Recommended Cleaning Actions ({len(recs)} suggestions)", expanded=True):
            for r in recs:
                p_badge = "🔴 High" if r["priority"] == "High" else ("🟡 Medium" if r["priority"] == "Medium" else "🟢 Low")
                st.markdown(f"**{p_badge} Priority — {r['action']}**")
                st.write(r["description"])
                st.caption(f"Suggested remedy: {r['suggested_fix']}")
                st.divider()

    # 3. Interactive Cleaning Tabs
    tab_missing, tab_dups, tab_types, tab_outliers, tab_text, tab_cols, tab_history = st.tabs([
        "1. Missing Values",
        "2. Duplicates",
        "3. Data Types",
        "4. Outliers",
        "5. Text Hygiene",
        "6. Columns Management",
        "7. Audit Log & Revert"
    ])

    # -------------------------------------------------------------
    # TAB 1: MISSING VALUES
    # -------------------------------------------------------------
    with tab_missing:
        st.markdown("#### Handle Missing Data")
        m_col1, m_col2 = st.columns(2)

        with m_col1:
            st.markdown("##### 💉 Numerical Imputation")
            num_cols_missing = [c for c in df.select_dtypes(include=["number"]).columns if df[c].isna().sum() > 0]
            if num_cols_missing:
                target_num = st.selectbox("Select Numeric Column", options=num_cols_missing, key="impute_num_col")
                strat = st.selectbox("Imputation Strategy", options=["median", "mean", "zero", "ffill", "bfill"], key="num_strat")
                if st.button("Apply Numerical Imputation", key="btn_imp_num"):
                    cleaning_engine.impute_missing_numerical(target_num, strategy=strat)
                    _sync_cleaned_state(cleaning_engine)
                    st.success(f"Imputed missing values in '{target_num}' using {strat}!")
                    st.rerun()
            else:
                st.info("No numerical columns currently have missing values.")

        with m_col2:
            st.markdown("##### 🏷️ Categorical Imputation")
            cat_cols_missing = [c for c in df.select_dtypes(exclude=["number"]).columns if df[c].isna().sum() > 0]
            if cat_cols_missing:
                target_cat = st.selectbox("Select Categorical Column", options=cat_cols_missing, key="impute_cat_col")
                cat_strat = st.selectbox("Strategy", options=["mode", "custom", "Missing"], key="cat_strat")
                custom_text = None
                if cat_strat == "custom":
                    custom_text = st.text_input("Custom Replacement Value", value="Unknown", key="custom_cat_val")
                if st.button("Apply Categorical Imputation", key="btn_imp_cat"):
                    cleaning_engine.impute_missing_categorical(target_cat, strategy=cat_strat, custom_val=custom_text)
                    _sync_cleaned_state(cleaning_engine)
                    st.success(f"Imputed missing values in '{target_cat}'!")
                    st.rerun()
            else:
                st.info("No categorical columns currently have missing values.")

        st.markdown("---")
        st.markdown("##### 🗑️ Drop Rows or High-Missing Columns")
        d_c1, d_c2 = st.columns(2)
        with d_c1:
            if st.button("Drop Rows with ANY Missing Values", key="btn_drop_rows"):
                cleaning_engine.drop_missing_rows(how="any")
                _sync_cleaned_state(cleaning_engine)
                st.success("Dropped rows with missing values!")
                st.rerun()
        with d_c2:
            thresh = st.slider("Missingness Threshold (%) to Drop Column", min_value=20, max_value=90, value=50, step=5)
            if st.button(f"Drop Columns with > {thresh}% Missing", key="btn_drop_cols"):
                cleaning_engine.drop_missing_columns(threshold_pct=thresh)
                _sync_cleaned_state(cleaning_engine)
                st.success(f"Dropped columns exceeding {thresh}% missing values!")
                st.rerun()

    # -------------------------------------------------------------
    # TAB 2: DUPLICATES
    # -------------------------------------------------------------
    with tab_dups:
        st.markdown("#### Deduplication")
        dup_count = int(df.duplicated().sum())
        st.write(f"Detected **{dup_count:,} duplicate rows** ({dup_count/len(df)*100:.2f}% of dataset).")

        if dup_count > 0:
            st.dataframe(df[df.duplicated(keep=False)].head(10), width='stretch')
            keep_opt = st.radio("Keep Strategy", options=["first", "last"], horizontal=True, key="dup_keep")
            if st.button("🚨 Remove All Duplicates", type="primary", key="btn_rem_dups"):
                cleaning_engine.remove_duplicates(keep=keep_opt)
                _sync_cleaned_state(cleaning_engine)
                st.success(f"Removed {dup_count} duplicate rows!")
                st.rerun()
        else:
            st.success("✅ Zero duplicate rows detected in the dataset!")

    # -------------------------------------------------------------
    # TAB 3: DATA TYPES
    # -------------------------------------------------------------
    with tab_types:
        st.markdown("#### Type Standardization")
        t_col1, t_col2 = st.columns(2)

        with t_col1:
            st.markdown("##### 💵 Clean Formatted Currency / Percent Strings")
            string_cols = df.select_dtypes(include=["object"]).columns.tolist()
            if string_cols:
                target_str = st.selectbox("Select String Column to Convert to Float", options=string_cols, key="num_str_col")
                if st.button("Convert to Clean Numeric", key="btn_conv_num"):
                    cleaning_engine.convert_to_numeric_clean(target_str)
                    _sync_cleaned_state(cleaning_engine)
                    st.success(f"Converted '{target_str}' to float!")
                    st.rerun()

        with t_col2:
            st.markdown("##### 📅 Parse Datetime")
            if string_cols:
                target_date = st.selectbox("Select Date Column to Parse", options=string_cols, key="date_str_col")
                if st.button("Convert to Datetime", key="btn_conv_date"):
                    cleaning_engine.convert_to_datetime(target_date)
                    _sync_cleaned_state(cleaning_engine)
                    st.success(f"Parsed '{target_date}' to datetime!")
                    st.rerun()

    # -------------------------------------------------------------
    # TAB 4: OUTLIERS
    # -------------------------------------------------------------
    with tab_outliers:
        st.markdown("#### Outlier Detection & Treatment")
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if num_cols:
            outlier_target = st.selectbox("Select Numeric Column to Audit", options=num_cols, key="outlier_col")
            method = st.radio("Detection Method", options=["IQR Method (Standard)", "Z-Score Method (Parametric)"], horizontal=True)
            method_key = "iqr" if "IQR" in method else "zscore"

            if method_key == "iqr":
                mask, lower, upper, count = cleaning_engine.detect_outliers_iqr(outlier_target, multiplier=1.5)
            else:
                mask, lower, upper, count = cleaning_engine.detect_outliers_zscore(outlier_target, threshold=3.0)

            st.write(f"Detected **{count} outliers** in `{outlier_target}` outside acceptable range `[{lower:.2f}, {upper:.2f}]`.")

            if count > 0:
                st.dataframe(df[mask].head(10), width='stretch')
                act = st.radio("Treatment Action", options=["Cap / Winsorize (Clamp to boundaries)", "Drop Outlier Rows"], horizontal=True)
                act_key = "cap" if "Cap" in act else "drop"

                if st.button("Apply Outlier Remediation", type="primary", key="btn_outlier"):
                    cleaning_engine.handle_outliers(outlier_target, method=method_key, action=act_key)
                    _sync_cleaned_state(cleaning_engine)
                    st.success(f"Remediated {count} outliers in '{outlier_target}'!")
                    st.rerun()
            else:
                st.success(f"No statistical outliers detected in '{outlier_target}'.")

    # -------------------------------------------------------------
    # TAB 5: TEXT HYGIENE
    # -------------------------------------------------------------
    with tab_text:
        st.markdown("#### Text Standardization & Whitespace Trimming")
        str_cols = df.select_dtypes(include=["object"]).columns.tolist()
        if str_cols:
            text_target = st.selectbox("Select Text Column", options=str_cols, key="text_col")
            case_opt = st.selectbox("Casing Format", options=["title", "lower", "upper"], key="text_case")
            strip_ws = st.checkbox("Strip Leading and Trailing Whitespace", value=True, key="strip_ws")

            if st.button("Apply Text Normalization", key="btn_text"):
                cleaning_engine.standardize_text(text_target, case=case_opt, strip_whitespace=strip_ws)
                _sync_cleaned_state(cleaning_engine)
                st.success(f"Standardized text in '{text_target}'!")
                st.rerun()

    # -------------------------------------------------------------
    # TAB 6: COLUMNS MANAGEMENT
    # -------------------------------------------------------------
    with tab_cols:
        st.markdown("#### Rename or Drop Columns")
        c_ren, c_drop = st.columns(2)

        with c_ren:
            st.markdown("##### ✏️ Rename Column")
            old_col = st.selectbox("Choose Column to Rename", options=df.columns.tolist(), key="ren_old")
            new_col = st.text_input("New Name", value=old_col, key="ren_new")
            if st.button("Rename Column", key="btn_ren"):
                if new_col and new_col != old_col:
                    cleaning_engine.rename_column(old_col, new_col)
                    _sync_cleaned_state(cleaning_engine)
                    st.success(f"Renamed '{old_col}' to '{new_col}'!")
                    st.rerun()

        with c_drop:
            st.markdown("##### 🗑️ Drop Columns")
            drop_targets = st.multiselect("Select Columns to Drop", options=df.columns.tolist(), key="drop_targets")
            if drop_targets and st.button("Drop Selected Columns", type="primary", key="btn_drop_multi"):
                cleaning_engine.drop_columns(drop_targets)
                _sync_cleaned_state(cleaning_engine)
                st.success(f"Dropped {len(drop_targets)} columns!")
                st.rerun()

    # -------------------------------------------------------------
    # TAB 7: AUDIT LOG & REVERT
    # -------------------------------------------------------------
    with tab_history:
        st.markdown("#### 📜 Cleaning Operation Audit Log")
        log = cleaning_engine.get_log()
        if log:
            st.dataframe(pd.DataFrame(log), width='stretch', hide_index=True)
        else:
            st.info("No cleaning operations have been applied yet.")

        st.markdown("---")
        st.markdown("#### ↩️ Revert to Original Dataset")
        st.warning("This will discard all cleaning transformations and restore the raw uploaded state.")
        if st.button("Revert All Changes to Original", type="secondary", key="btn_revert"):
            cleaning_engine.reset_to_original()
            _sync_cleaned_state(cleaning_engine)
            st.success("Reverted to original dataset state!")
            st.rerun()


def _sync_cleaned_state(engine: CleaningEngine):
    """Synchronize cleaned dataframe and re-run dependent analytical profiling."""
    cleaned = engine.get_cleaned_df()
    st.session_state["df_cleaned"] = cleaned

    # Re-profile
    roles = st.session_state.get("column_roles", {})
    profile = DataProfiler.profile_dataset(cleaned, column_roles=roles)
    st.session_state["profile_data"] = profile

    # Re-compute KPIs & Insights
    domain = st.session_state.get("domain", "general")
    st.session_state["kpis"] = KPIEngine.compute_kpis(cleaned, roles, domain=domain)
    st.session_state["insights"] = InsightEngine().generate_all_insights(cleaned, roles, domain=domain)

