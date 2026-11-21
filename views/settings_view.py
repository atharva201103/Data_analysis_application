"""
Settings and Configuration View for DataSense AI.
Configures modular LLM providers (Ollama, OpenAI, Groq, Gemini) and semantic column mappings.
"""

import os
import streamlit as st
import pandas as pd
from utils.constants import (
    LLM_PROVIDERS, PROVIDER_OLLAMA, PROVIDER_OPENAI, PROVIDER_GROQ,
    PROVIDER_GEMINI, PROVIDER_ANTHROPIC, SEMANTIC_ROLES
)
from utils.ui_components import render_page_header, render_section_header
from backend.ai_interpreter import AIInterpreter
from backend.column_detector import ColumnDetector
from backend.kpi_engine import KPIEngine
from backend.insight_engine import InsightEngine


def render_settings_view():
    render_page_header("Application & AI Settings", "Configure LLM providers, semantic column mappings, and system preferences", icon="⚙️", eyebrow="Settings")

    tab_llm, tab_mapping, tab_app = st.tabs(["1. LLM Provider Configuration", "2. Semantic Role Mapping", "3. Application Preferences"])

    # -------------------------------------------------------------
    # TAB 1: LLM PROVIDERS
    # -------------------------------------------------------------
    with tab_llm:
        st.markdown("### 🤖 Artificial Intelligence Provider")
        st.markdown(
            """
            DataSense AI operates with a modular AI interpretation engine. 
            Connect local **Ollama** for private on-device processing, or cloud APIs (**OpenAI, Groq, Gemini**).
            *Note: The platform is fully functional in non-AI mode using our deterministic business rule engine.*
            """
        )

        curr_provider = st.session_state.get("llm_provider", os.getenv("LLM_PROVIDER", PROVIDER_GEMINI)).lower()
        provider_idx = LLM_PROVIDERS.index(curr_provider) if curr_provider in LLM_PROVIDERS else LLM_PROVIDERS.index(PROVIDER_GEMINI)

        selected_provider = st.selectbox("Select LLM Provider", options=LLM_PROVIDERS, index=provider_idx)

        c1, c2 = st.columns(2)
        with c1:
            default_model = (
                "gemini-2.5-flash-lite" if selected_provider == PROVIDER_GEMINI else (
                    "gpt-4o-mini" if selected_provider == PROVIDER_OPENAI else (
                        "llama-3.1-70b-versatile" if selected_provider == PROVIDER_GROQ else (
                            "claude-3-haiku-20240307" if selected_provider == PROVIDER_ANTHROPIC else "llama3:8b"
                        )
                    )
                )
            )
            model_name = st.text_input(
                "Model Identifier",
                value=st.session_state.get("llm_model", os.getenv("LLM_MODEL", default_model)),
                help="Name of the model to invoke."
            )

        with c2:
            if selected_provider == PROVIDER_OLLAMA:
                base_url = st.text_input(
                    "Ollama Server URL",
                    value=st.session_state.get("llm_base_url", os.getenv("LLM_BASE_URL", "http://localhost:11434")),
                    help="Default Ollama endpoint is http://localhost:11434"
                )
                api_key = ""
            else:
                base_url = ""
                api_key = st.text_input(
                    f"{selected_provider.upper()} API Key",
                    value=st.session_state.get("llm_api_key", os.getenv(f"{selected_provider.upper()}_API_KEY", "")),
                    type="password",
                    help="API key is stored only in session state and never written to disk."
                )

        c_save, c_test = st.columns(2)
        with c_save:
            if st.button("💾 Save AI Settings", type="primary", width='stretch'):
                st.session_state["llm_provider"] = selected_provider
                st.session_state["llm_model"] = model_name
                st.session_state["llm_base_url"] = base_url
                st.session_state["llm_api_key"] = api_key
                st.success("AI configuration saved for current session!")

        with c_test:
            if st.button("🔌 Test Connection", width='stretch'):
                test_interpreter = AIInterpreter(
                    provider=selected_provider,
                    model=model_name,
                    api_key=api_key,
                    base_url=base_url
                )
                with st.spinner(f"Testing connection to {selected_provider.upper()}..."):
                    if test_interpreter.is_ai_available():
                        st.success(f"✅ Connection successful! {selected_provider.upper()} is active and reachable.")
                    else:
                        st.error(f"❌ Could not establish live connection to {selected_provider.upper()}. Check endpoint or API key.")

    # -------------------------------------------------------------
    # TAB 2: SEMANTIC ROLE MAPPING (ADAPTABILITY)
    # -------------------------------------------------------------
    with tab_mapping:
        st.markdown("### 🏷️ Semantic Column Role Mapping")
        st.markdown(
            """
            DataSense AI automatically infers whether columns represent Revenue, Cost, Profit, Date, 
            Category, or other business roles. You can manually adjust or override mappings below:
            """
        )

        df = st.session_state.get("df_cleaned")
        if df is not None:
            column_roles = st.session_state.get("column_roles", {})
            overrides = st.session_state.get("column_role_overrides", {})

            col_map_inputs = {}
            m_cols = st.columns(3)

            for idx, col_name in enumerate(df.columns):
                current_role = overrides.get(col_name, column_roles.get(col_name, {}).get("role", "General"))
                role_idx = SEMANTIC_ROLES.index(current_role) if current_role in SEMANTIC_ROLES else 0

                with m_cols[idx % 3]:
                    selected_role = st.selectbox(
                        f"Column: {col_name}",
                        options=SEMANTIC_ROLES,
                        index=role_idx,
                        key=f"role_map_{col_name}"
                    )
                    col_map_inputs[col_name] = selected_role

            if st.button("Apply & Re-compute Role Mappings", type="primary"):
                st.session_state["column_role_overrides"] = col_map_inputs
                # Re-run detector
                detector = ColumnDetector(overrides=col_map_inputs)
                new_roles = detector.detect_all(df)
                st.session_state["column_roles"] = new_roles

                # Re-run KPIs and Insights with updated roles
                domain = st.session_state.get("domain", "general")
                st.session_state["kpis"] = KPIEngine.compute_kpis(df, new_roles, domain=domain)
                st.session_state["insights"] = InsightEngine().generate_all_insights(df, new_roles, domain=domain)

                st.success("Successfully updated semantic column roles and re-calculated business metrics!")
                st.rerun()
        else:
            st.info("Upload or load a dataset to configure column role mappings.")

    # -------------------------------------------------------------
    # TAB 3: APP PREFERENCES
    # -------------------------------------------------------------
    with tab_app:
        st.markdown("### ⚙️ System Preferences & Session Reset")
        st.write("DataSense AI Version: `1.0.0`")

        st.markdown("---")
        st.markdown("#### 🗑️ Clear All Session Cache & Data")
        if st.button("Reset Session State", type="secondary"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.success("All session data has been wiped.")
            st.rerun()

