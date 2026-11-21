"""
Chat With Your Data View for DataSense AI.
Provides a conversational natural language interface with safe, validated analytical execution.
"""

import streamlit as st
import pandas as pd
from utils.ui_components import render_page_header, render_section_header
from backend.chat_engine import ChatEngine


def render_chat_view():
    df = st.session_state.get("df_cleaned")
    if df is None:
        st.warning("⚠️ No dataset loaded. Please upload a dataset in 'Upload Dataset' first.")
        return

    dataset_name = st.session_state.get("dataset_name", "Dataset")
    column_roles = st.session_state.get("column_roles", {})

    render_page_header("Chat With Your Data", f"Ask questions in plain English about: {dataset_name}", icon="💬", eyebrow="AI Chat Interface")

    # Initialize chat engine and message store
    chat_engine = ChatEngine(df, column_roles)
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []

    # Quick prompt chips
    st.markdown("##### 💡 Suggested Questions")
    chip_col1, chip_col2, chip_col3, chip_col4 = st.columns(4)

    prompt_to_run = None
    with chip_col1:
        if st.button("🏆 Highest Performer", width='stretch'):
            prompt_to_run = "Which category generated the highest revenue?"
    with chip_col2:
        if st.button("📊 Average Order Value", width='stretch'):
            prompt_to_run = "What is the average order value?"
    with chip_col3:
        if st.button("📈 Revenue Timeline", width='stretch'):
            prompt_to_run = "Show monthly revenue trends"
    with chip_col4:
        if st.button("📑 Executive Summary", width='stretch'):
            prompt_to_run = "Give me an overview of this dataset"

    # Display message history
    for msg in st.session_state["chat_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("table") is not None and not msg["table"].empty:
                st.dataframe(msg["table"], width='stretch', hide_index=True)
            if msg.get("figure") is not None:
                st.plotly_chart(msg["figure"], width='stretch')

    # Chat input
    user_query = st.chat_input("Ask a question about your dataset (e.g. 'Which region has the lowest profit?')...")
    if prompt_to_run:
        user_query = prompt_to_run

    if user_query:
        # Display user message
        st.session_state["chat_messages"].append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing dataset and computing answer..."):
                result = chat_engine.process_query(user_query)
                answer_text = result.get("answer", "No response.")
                data_table = result.get("data")
                figure = result.get("figure")

                st.markdown(answer_text)
                if data_table is not None and not data_table.empty:
                    st.dataframe(data_table, width='stretch', hide_index=True)
                if figure is not None:
                    st.plotly_chart(figure, width='stretch')

                st.session_state["chat_messages"].append({
                    "role": "assistant",
                    "content": answer_text,
                    "table": data_table,
                    "figure": figure
                })

    # Clear chat option
    if st.session_state["chat_messages"]:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑️ Clear Conversation History"):
            st.session_state["chat_messages"] = []
            st.rerun()

