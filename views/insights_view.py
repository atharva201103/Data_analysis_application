"""
Business Insights View for DataSense AI.
Presents structured 5-part executive business insights and strategic recommendations.
"""

import streamlit as st
import pandas as pd
from utils.ui_components import render_page_header, render_section_header, render_insight_card
from backend.insight_engine import InsightEngine
from backend.ai_interpreter import AIInterpreter


def render_insights_view():
    df = st.session_state.get("df_cleaned")
    if df is None:
        st.warning("⚠️ No dataset loaded. Please upload a dataset in 'Upload Dataset' first.")
        return

    dataset_name = st.session_state.get("dataset_name", "Dataset")
    column_roles = st.session_state.get("column_roles", {})
    domain = st.session_state.get("domain", "general")

    render_page_header("Intelligent Business Insights", f"Actionable strategic intelligence for: {dataset_name}", icon="💡", eyebrow="Business Intelligence")

    # AI Status Banner
    ai_interpreter = AIInterpreter(
        provider=st.session_state.get("llm_provider"),
        model=st.session_state.get("llm_model"),
        api_key=st.session_state.get("llm_api_key"),
        base_url=st.session_state.get("llm_base_url")
    )
    is_live_ai = ai_interpreter.is_ai_available()

    col_status, col_btn = st.columns([2.5, 1.0])
    with col_status:
        if is_live_ai:
            st.success(f"🤖 Connected to AI Provider: `{ai_interpreter.provider.upper()}` (Model: `{ai_interpreter.model}`)")
        else:
            st.info("⚡ Operating in Deterministic Rule Engine Mode (Zero API dependency). Connect an LLM in Settings if desired.")

    with col_btn:
        if st.button("🔄 Refresh All Insights", width='stretch'):
            with st.spinner("Re-evaluating business rules and generating insights..."):
                engine = InsightEngine(ai_interpreter=ai_interpreter)
                st.session_state["insights"] = engine.generate_all_insights(df, column_roles, domain=domain)
                st.success("Refreshed insights!")
                st.rerun()

    insights = st.session_state.get("insights", [])

    if not insights:
        st.info("No business rule triggers detected for this dataset structure. Make sure key column roles (e.g. Revenue, Category, Region) are mapped.")
        return

    # Filter by category
    categories = ["All Insights"] + sorted(list(set(ins.get("category", "general").capitalize() for ins in insights)))
    chosen_cat = st.radio("Filter Insights by Category", options=categories, horizontal=True)

    filtered_insights = insights
    if chosen_cat != "All Insights":
        filtered_insights = [ins for ins in insights if ins.get("category", "general").lower() == chosen_cat.lower()]

    st.markdown(f"### 📋 Showing {len(filtered_insights)} Strategic Findings")

    for ins in filtered_insights:
        render_insight_card(
            title=ins.get("title", "Insight"),
            observation=ins.get("observation", ""),
            implication=ins.get("implication", ""),
            recommendation=ins.get("recommendation", ""),
            metrics_summary=ins.get("evidence"),
            category=ins.get("category", "general")
        )

