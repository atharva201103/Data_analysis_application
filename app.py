"""
DataSense AI – Automated Data Analysis & Business Intelligence Platform
Primary Application Entry Point.
"""

import os
import streamlit as st
import pandas as pd

# Load .env file if present (before any other imports that read env vars)
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=False)
except ImportError:
    pass  # python-dotenv not installed; env vars still work if set externally

# Set matplotlib config dir to prevent warning
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from utils.constants import APP_NAME, APP_TAGLINE, APP_VERSION
from utils.ui_components import inject_custom_css, render_html_block
from views.dashboard_view import render_dashboard_view
from views.upload_view import render_upload_view
from views.profiling_view import render_profiling_view
from views.cleaning_view import render_cleaning_view
from views.eda_view import render_eda_view
from views.visualization_view import render_visualization_view
from views.insights_view import render_insights_view
from views.chat_view import render_chat_view
from views.reports_view import render_reports_view
from views.settings_view import render_settings_view


def main():
    # 1. Page Configuration
    st.set_page_config(
        page_title=f"{APP_NAME} – Automated Business Intelligence",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # 2. Inject Modern SaaS Theme CSS
    inject_custom_css()

    # 3. Initialize Default Session State
    if "nav_selection" not in st.session_state:
        st.session_state["nav_selection"] = "Dashboard"

    # 4. Modern Sidebar Navigation
    with st.sidebar:
        brand_html = (
            f'<div style="padding: 0.5rem 0 1.2rem 0; border-bottom: 1px solid rgba(255,255,255,0.08); margin-bottom: 1rem;">'
            f'<div style="font-size: 1.55rem; font-weight: 800; color: #F8FAFC; letter-spacing: -0.02em; display: flex; align-items: center; gap: 0.5rem;">'
            f'<span style="background: linear-gradient(135deg, #4F46E5, #06B6D4); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{APP_NAME}</span>'
            f'</div>'
            f'<div style="font-size: 0.78rem; color: #94A3B8; font-weight: 500; margin-top: 0.2rem;">'
            f'{APP_TAGLINE}'
            f'</div>'
            f'</div>'
        )
        render_html_block(brand_html)

        # Active Dataset Badge
        df_active = st.session_state.get("df_cleaned")
        if df_active is not None:
            dataset_name = st.session_state.get("dataset_name", "Active Dataset")
            health_score = st.session_state.get("profile_data", {}).get("health_score", 100)
            score_color = "#10B981" if health_score >= 80 else ("#F59E0B" if health_score >= 60 else "#EF4444")

            badge_html = (
                f'<div style="background: rgba(30, 41, 59, 0.8); border: 1px solid rgba(255,255,255,0.1); border-radius: 8px; padding: 0.7rem 0.9rem; margin-bottom: 1.2rem;">'
                f'<div style="font-size: 0.72rem; text-transform: uppercase; color: #94A3B8; font-weight: 600;">Active Dataset</div>'
                f'<div style="font-size: 0.88rem; font-weight: 700; color: #F8FAFC; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{dataset_name}</div>'
                f'<div style="font-size: 0.75rem; color: #64748B; margin-top: 0.2rem;">'
                f'{len(df_active):,} rows • {len(df_active.columns)} cols • <span style="color: {score_color}; font-weight: 600;">Health: {health_score}/100</span>'
                f'</div>'
                f'</div>'
            )
            render_html_block(badge_html)
        else:
            no_data_html = (
                '<div style="background: rgba(30, 41, 59, 0.4); border: 1px dashed rgba(255,255,255,0.1); border-radius: 8px; padding: 0.6rem 0.8rem; margin-bottom: 1.2rem; text-align: center;">'
                '<span style="font-size: 0.8rem; color: #94A3B8;">No dataset loaded</span>'
                '</div>'
            )
            render_html_block(no_data_html)

        # Navigation Menu
        nav_options = [
            "Dashboard",
            "Upload Dataset",
            "Data Profiling",
            "Data Cleaning",
            "Exploratory Data Analysis",
            "Data Visualization",
            "Business Insights",
            "Chat With Your Data",
            "Export Reports",
            "Settings"
        ]

        nav_icons = {
            "Dashboard": "📊",
            "Upload Dataset": "📁",
            "Data Profiling": "🔍",
            "Data Cleaning": "🧹",
            "Exploratory Data Analysis": "📈",
            "Data Visualization": "📉",
            "Business Insights": "💡",
            "Chat With Your Data": "💬",
            "Export Reports": "📑",
            "Settings": "⚙️"
        }

        # Index lookup
        current_selection = st.session_state.get("nav_selection", "Dashboard")
        default_idx = nav_options.index(current_selection) if current_selection in nav_options else 0

        selection = st.radio(
            "Navigation Menu",
            options=nav_options,
            format_func=lambda x: f"{nav_icons.get(x, '•')}  {x}",
            index=default_idx,
            label_visibility="collapsed"
        )
        st.session_state["nav_selection"] = selection

        # Quick sample load button if no dataset
        if df_active is None:
            st.markdown("---")
            if st.button("⚡ Quick Load Sample", width='stretch'):
                from backend.data_loader import DataLoader
                from views.upload_view import _initialize_dataset
                sample_file = "sample_datasets/retail_sales_sample.csv"
                df, _ = DataLoader.load_dataset(sample_file, "retail_sales_sample.csv")
                _initialize_dataset(df, "retail_sales_sample.csv")
                st.session_state["nav_selection"] = "Dashboard"
                st.rerun()

        # Sidebar Footer
        footer_html = (
            f'<div style="margin-top: 3rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.06); font-size: 0.72rem; color: #64748B; text-align: center;">'
            f'{APP_NAME} v{APP_VERSION}<br>Enterprise BI & Analytical Engine'
            f'</div>'
        )
        render_html_block(footer_html)

    # 5. Route to Selected View
    if selection == "Dashboard":
        render_dashboard_view()
    elif selection == "Upload Dataset":
        render_upload_view()
    elif selection == "Data Profiling":
        render_profiling_view()
    elif selection == "Data Cleaning":
        render_cleaning_view()
    elif selection == "Exploratory Data Analysis":
        render_eda_view()
    elif selection == "Data Visualization":
        render_visualization_view()
    elif selection == "Business Insights":
        render_insights_view()
    elif selection == "Chat With Your Data":
        render_chat_view()
    elif selection == "Export Reports":
        render_reports_view()
    elif selection == "Settings":
        render_settings_view()


if __name__ == "__main__":
    main()

