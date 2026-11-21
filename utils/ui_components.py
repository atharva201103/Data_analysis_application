"""
UI components and CSS injection for DataSense AI.
Provides a modern SaaS analytics dashboard look and feel with Lucide vector icons.
"""

import textwrap
from typing import Optional, List, Dict, Any
import streamlit as st


# =============================================================================
# Lucide Icon SVG Definitions
# =============================================================================
LUCIDE_SVG_BODIES = {
    "dollar-sign": '<line x1="12" y1="1" x2="12" y2="23"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/>',
    "trending-up": '<polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/>',
    "trending-down": '<polyline points="23 18 13.5 8.5 8.5 13.5 1 6"/><polyline points="17 18 23 18 23 12"/>',
    "shopping-cart": '<circle cx="8" cy="21" r="1"/><circle cx="19" cy="21" r="1"/><path d="M2.05 2.05h2l2.66 12.42a2 2 0 0 0 2 1.58h9.78a2 2 0 0 0 1.95-1.57l1.65-7.43H5.12"/>',
    "package": '<path d="m7.5 4.27 9 5.15"/><path d="M21 8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16Z"/><path d="m3.3 7 8.7 5 8.7-5"/><path d="M12 22V12"/>',
    "users": '<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>',
    "user": '<path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>',
    "briefcase": '<rect width="20" height="14" x="2" y="7" rx="2" ry="2"/><path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"/>',
    "percent": '<line x1="19" y1="5" x2="5" y2="19"/><circle cx="6.5" cy="6.5" r="2.5"/><circle cx="17.5" cy="17.5" r="2.5"/>',
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "star": '<polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>',
    "user-minus": '<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><line x1="22" y1="11" x2="16" y2="11"/>',
    "database": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5V19A9 3 0 0 0 21 19V5"/><path d="M3 12A9 3 0 0 0 21 12"/>',
    "columns": '<rect width="18" height="18" x="3" y="3" rx="2"/><path d="M12 3v18"/>',
    "alert-triangle": '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
    "copy": '<rect width="14" height="14" x="8" y="8" rx="2" ry="2"/><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/>',
    "hard-drive": '<line x1="22" y1="12" x2="2" y2="12"/><path d="M5.45 5.11 2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11z"/><line x1="6" y1="16" x2="6.01" y2="16"/><line x1="10" y1="16" x2="10.01" y2="16"/>',
    "check-circle": '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>',
    "zap": '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>',
    "activity": '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>',
    "bar-chart-2": '<line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/>',
    "file-text": '<path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><line x1="10" y1="9" x2="8" y2="9"/>'
}

# Mapping legacy emojis or synonyms to clean Lucide icons
ICON_ALIASES = {
    "💰": "dollar-sign",
    "💵": "dollar-sign",
    "revenue": "dollar-sign",
    "sales": "dollar-sign",
    "profit": "trending-up",
    "📈": "trending-up",
    "growth": "trending-up",
    "📉": "trending-down",
    "loss": "trending-down",
    "🛒": "shopping-cart",
    "orders": "shopping-cart",
    "aov": "shopping-cart",
    "📦": "package",
    "volume": "package",
    "units": "package",
    "👥": "users",
    "headcount": "users",
    "employees": "users",
    "👤": "user",
    "customers": "users",
    "💼": "briefcase",
    "salary": "briefcase",
    "🎯": "percent",
    "margin": "percent",
    "discount": "percent",
    "⭐": "star",
    "satisfaction": "star",
    "🚪": "user-minus",
    "attrition": "user-minus",
    "churn": "user-minus",
    "🔢": "database",
    "rows": "database",
    "records": "database",
    "📊": "bar-chart-2",
    "columns": "columns",
    "chart": "bar-chart-2",
    "⚠️": "alert-triangle",
    "warning": "alert-triangle",
    "missing": "alert-triangle",
    "💾": "hard-drive",
    "memory": "hard-drive",
    "✅": "check-circle",
    "resolved": "check-circle",
    "⚡": "zap",
    "operations": "zap",
    "📑": "file-text",
    "report": "file-text",
    "💡": "zap",
    "🚀": "trending-up"
}


def get_lucide_icon(
    icon_name: Optional[str] = None,
    size: int = 16,
    color: str = "currentColor",
    stroke_width: int = 2
) -> str:
    """Return an inline SVG string for a Lucide icon."""
    key = str(icon_name or "bar-chart-2").strip().lower()
    canonical_key = ICON_ALIASES.get(key, key)
    body = LUCIDE_SVG_BODIES.get(canonical_key, LUCIDE_SVG_BODIES["bar-chart-2"])
    
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke_width}" '
        f'stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-{canonical_key}" '
        f'style="vertical-align: middle; flex-shrink: 0;">{body}</svg>'
    )


def render_html_block(html_content: str):
    """Safely render an HTML block without markdown compiler interference."""
    clean = textwrap.dedent(html_content).strip()
    if hasattr(st, "html"):
        st.html(clean)
    else:
        st.markdown(clean, unsafe_allow_html=True)


# =============================================================================
# Modern SaaS CSS Injection
# =============================================================================
def inject_custom_css():
    """Inject modern SaaS styling into the Streamlit app."""
    css = """
    <style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* =========================================================
       GLOBAL LAYOUT — push ALL page content below Streamlit
       toolbar (≈ 3.75 rem). Use clamp() so it scales on small
       screens while never collapsing below 3 rem.
       ========================================================= */
    .block-container {
        padding-top: clamp(3rem, 4.5vw, 4.5rem) !important;
        padding-bottom: 3rem !important;
        padding-left: clamp(1rem, 2.5vw, 2.5rem) !important;
        padding-right: clamp(1rem, 2.5vw, 2.5rem) !important;
        max-width: 1400px;
        overflow-x: hidden;
    }

    /* Prevent any stMain wrapper from adding extra top offsets */
    [data-testid="stAppViewContainer"] > section > div:first-child {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }

    /* Remove Streamlit's own top padding override that sometimes
       sets it very low on rerun (Streamlit ≥ 1.33 injects inline
       style="padding-top: 0px" on some builds) */
    section[data-testid="stMain"] > div {
        padding-top: 0 !important;
    }

    /* Card Styling */
    .ds-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.2rem;
        backdrop-filter: blur(8px);
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.25);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .ds-card:hover {
        border-color: rgba(99, 102, 241, 0.3);
    }

    /* KPI Metric Card */
    .kpi-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.85) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(99, 102, 241, 0.22);
        border-radius: 14px;
        padding: 1.15rem 1.35rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.35);
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
        margin-bottom: 0.8rem;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 15px 30px -5px rgba(79, 70, 229, 0.22);
        border-color: rgba(99, 102, 241, 0.5);
    }
    .kpi-header {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        margin-bottom: 0.5rem;
    }
    .kpi-icon-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 28px;
        height: 28px;
        border-radius: 7px;
        background: rgba(99, 102, 241, 0.12);
        color: #818CF8;
        border: 1px solid rgba(99, 102, 241, 0.25);
        flex-shrink: 0;
    }
    .kpi-title {
        color: #94A3B8;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .kpi-value {
        color: #F8FAFC;
        font-size: 1.85rem;
        font-weight: 700;
        line-height: 1.2;
        margin-bottom: 0.35rem;
        font-family: 'Inter', sans-serif;
    }
    .kpi-footer {
        display: flex;
        justify-content: space-between;
        align-items: center;
        min-height: 20px;
    }
    .kpi-subtext {
        color: #64748B;
        font-size: 0.76rem;
        font-weight: 500;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .kpi-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.18rem 0.55rem;
        border-radius: 9999px;
        font-size: 0.72rem;
        font-weight: 600;
        white-space: nowrap;
    }
    .kpi-badge.positive {
        background: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .kpi-badge.negative {
        background: rgba(239, 68, 68, 0.15);
        color: #EF4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }

    /* Health Score Dial */
    .health-score-container {
        text-align: center;
        padding: 1.5rem;
        background: radial-gradient(circle at center, rgba(79, 70, 229, 0.12) 0%, rgba(15, 23, 42, 0.6) 70%);
        border-radius: 16px;
        border: 1px solid rgba(99, 102, 241, 0.25);
    }
    .health-score-val {
        font-size: 3.2rem;
        font-weight: 800;
        line-height: 1;
        margin-bottom: 0.3rem;
    }
    .score-grade-excellent { color: #10B981; }
    .score-grade-good { color: #06B6D4; }
    .score-grade-fair { color: #F59E0B; }
    .score-grade-poor { color: #EF4444; }

    /* Tag Badges */
    .ds-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.25rem 0.65rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.35rem;
        margin-bottom: 0.35rem;
    }
    .badge-date { background: rgba(59, 130, 246, 0.15); color: #60A5FA; border: 1px solid rgba(59, 130, 246, 0.3); }
    .badge-revenue { background: rgba(16, 185, 129, 0.15); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.3); }
    .badge-cost { background: rgba(245, 158, 11, 0.15); color: #FBBF24; border: 1px solid rgba(245, 158, 11, 0.3); }
    .badge-category { background: rgba(139, 92, 246, 0.15); color: #A78BFA; border: 1px solid rgba(139, 92, 246, 0.3); }
    .badge-id { background: rgba(100, 116, 139, 0.15); color: #94A3B8; border: 1px solid rgba(100, 116, 139, 0.3); }

    /* Insight Card */
    .insight-box {
        background: rgba(30, 41, 59, 0.7);
        border-left: 4px solid #4F46E5;
        border-radius: 0 10px 10px 0;
        padding: 1rem 1.3rem;
        margin-bottom: 1rem;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    .insight-box.high-priority {
        border-left-color: #EF4444;
        background: rgba(239, 68, 68, 0.04);
    }
    .insight-box.opportunity {
        border-left-color: #10B981;
        background: rgba(16, 185, 129, 0.04);
    }
    .insight-box.warning {
        border-left-color: #F59E0B;
        background: rgba(245, 158, 11, 0.04);
    }

    /* Subdued Monospace Text */
    .code-pill {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        background: rgba(15, 23, 42, 0.6);
        padding: 0.15rem 0.4rem;
        border-radius: 4px;
        color: #E2E8F0;
    }

    /* =========================================================
       PAGE HEADER — large prominent heading used as the FIRST
       element on every view.  Provides guaranteed top clearance
       and a visual separator from subsequent content.
       ========================================================= */
    .page-header-wrap {
        padding-top: 0.25rem;
        padding-bottom: 1.5rem;
        margin-bottom: 0.5rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.07);
    }
    .page-header-eyebrow {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #818CF8;
        margin-bottom: 0.4rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    .page-header-title {
        font-size: clamp(1.5rem, 2.5vw, 2rem);
        font-weight: 800;
        color: #F8FAFC;
        letter-spacing: -0.02em;
        line-height: 1.2;
        margin-bottom: 0.4rem;
        display: flex;
        align-items: center;
        gap: 0.55rem;
    }
    .page-header-subtitle {
        font-size: 0.9rem;
        color: #94A3B8;
        font-weight: 400;
        line-height: 1.5;
        max-width: 720px;
    }

    /* =========================================================
       SECTION HEADERS — used for sub-sections within a page.
       Smaller than page-header; kept as-is for all existing
       render_section_header() callsites.
       ========================================================= */
    .section-header {
        font-size: 1.25rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-top: 1.2rem;
        margin-bottom: 0.3rem;
        display: flex;
        align-items: center;
        gap: 0.55rem;
    }
    .section-subtitle {
        color: #94A3B8;
        font-size: 0.88rem;
        margin-bottom: 1.2rem;
    }
    </style>
    """
    render_html_block(css)


# =============================================================================
# Component Renderers
# =============================================================================
def render_kpi_card(
    title: str,
    value: str,
    subtext: Optional[str] = None,
    delta: Optional[str] = None,
    delta_positive: bool = True,
    icon: str = "bar-chart-2"
):
    """
    Render a modern SaaS KPI card with Lucide vector icons.
    Strictly avoids nested indentation or unclosed tags to prevent literal HTML rendering.
    """
    svg_icon = get_lucide_icon(icon, size=16, color="#818CF8")

    delta_html = ""
    if delta:
        badge_class = "positive" if delta_positive else "negative"
        sign = "+" if delta_positive and not str(delta).startswith("+") else ""
        delta_html = f'<span class="kpi-badge {badge_class}">{sign}{delta}</span>'

    subtext_html = f'<span class="kpi-subtext">{subtext}</span>' if subtext else '<span class="kpi-subtext"></span>'

    card_html = (
        f'<div class="kpi-card">'
        f'<div class="kpi-header">'
        f'<span class="kpi-icon-badge">{svg_icon}</span>'
        f'<span class="kpi-title">{title}</span>'
        f'</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-footer">'
        f'{subtext_html}'
        f'{delta_html}'
        f'</div>'
        f'</div>'
    )
    render_html_block(card_html)


def render_health_score_card(score: int, grade: str, explanation: str):
    """Render an automated dataset health score widget."""
    color_class = "score-grade-poor"
    if score >= 85:
        color_class = "score-grade-excellent"
    elif score >= 70:
        color_class = "score-grade-good"
    elif score >= 50:
        color_class = "score-grade-fair"

    shield_icon = get_lucide_icon("activity", size=18, color="#818CF8")

    html = (
        f'<div class="health-score-container">'
        f'<div style="font-size: 0.85rem; font-weight: 600; text-transform: uppercase; color: #94A3B8; margin-bottom: 0.5rem; display: flex; align-items: center; justify-content: center; gap: 0.45rem;">'
        f'{shield_icon} <span>Dataset Quality Health Score</span>'
        f'</div>'
        f'<div class="health-score-val {color_class}">'
        f'{score}<span style="font-size: 1.5rem; color: #64748B;">/100</span>'
        f'</div>'
        f'<div style="font-size: 1.1rem; font-weight: 700; margin-bottom: 0.4rem; color: #F8FAFC;">'
        f'Grade: {grade}'
        f'</div>'
        f'<div style="font-size: 0.82rem; color: #94A3B8; max-width: 450px; margin: 0 auto; line-height: 1.4;">'
        f'{explanation}'
        f'</div>'
        f'</div>'
    )
    render_html_block(html)


def render_section_header(title: str, subtitle: Optional[str] = None, icon: str = ""):
    """Render a consistent section header with optional Lucide icon."""
    icon_html = f'<span style="color: #818CF8; display: inline-flex; align-items: center;">{get_lucide_icon(icon, size=22)}</span> ' if icon and (icon in LUCIDE_SVG_BODIES or icon in ICON_ALIASES) else (f"{icon} " if icon else "")
    full_html = (
        f'<div class="section-header">{icon_html}<span>{title}</span></div>'
        + (f'<div class="section-subtitle">{subtitle}</div>' if subtitle else '')
    )
    render_html_block(full_html)


def render_page_header(title: str, subtitle: Optional[str] = None, icon: str = "", eyebrow: Optional[str] = None):
    """
    Render a full-width page header as the FIRST element on a view.

    Provides guaranteed top clearance, large bold title, optional eyebrow
    label, subtitle, and a bottom separator — consistent across every page.

    Args:
        title:    Main page heading text.
        subtitle: Short description shown below the title.
        icon:     Lucide icon name, emoji, or any string shown left of title.
        eyebrow:  Small ALL-CAPS label above the title (e.g. "Analytics Module").
    """
    # Resolve icon to either Lucide SVG or plain text
    if icon and (icon in LUCIDE_SVG_BODIES or icon in ICON_ALIASES):
        icon_html = f'<span style="color:#818CF8;display:inline-flex;align-items:center;flex-shrink:0;">{get_lucide_icon(icon, size=28, color="#818CF8")}</span>'
    elif icon:
        icon_html = f'<span style="font-size:1.6rem;line-height:1;">{icon}</span>'
    else:
        icon_html = ""

    eyebrow_html = ""
    if eyebrow:
        eyebrow_html = (
            f'<div class="page-header-eyebrow">'
            f'<span style="width:3px;height:12px;background:#4F46E5;border-radius:2px;display:inline-block;"></span>'
            f'<span>{eyebrow}</span>'
            f'</div>'
        )

    subtitle_html = f'<div class="page-header-subtitle">{subtitle}</div>' if subtitle else ""

    html = (
        f'<div class="page-header-wrap">'
        f'{eyebrow_html}'
        f'<div class="page-header-title">'
        f'{icon_html}'
        f'<span>{title}</span>'
        f'</div>'
        f'{subtitle_html}'
        f'</div>'
    )
    render_html_block(html)


def render_insight_card(
    title: str,
    observation: str,
    implication: str,
    recommendation: str,
    metrics_summary: Optional[str] = None,
    category: str = "general"
):
    """Render a structured 5-part business insight card with Lucide vector icons."""
    box_class = "insight-box"
    cat_lower = category.lower()
    if cat_lower in ["risk", "danger", "decline", "warning"]:
        box_class += " high-priority"
        icon_name = "alert-triangle"
        icon_color = "#EF4444"
    elif cat_lower in ["opportunity", "growth", "top_performer"]:
        box_class += " opportunity"
        icon_name = "trending-up"
        icon_color = "#10B981"
    else:
        icon_name = "zap"
        icon_color = "#818CF8"

    lucide_svg = get_lucide_icon(icon_name, size=18, color=icon_color)

    metrics_html = ""
    if metrics_summary:
        metrics_html = (
            f'<div style="margin-top: 0.5rem; padding-top: 0.5rem; border-top: 1px dashed rgba(255,255,255,0.1); font-size: 0.8rem; color: #818CF8;">'
            f'<strong>Underlying Evidence:</strong> {metrics_summary}'
            f'</div>'
        )

    html = (
        f'<div class="{box_class}">'
        f'<div style="font-size: 1.05rem; font-weight: 700; color: #F8FAFC; margin-bottom: 0.45rem; display: flex; align-items: center; gap: 0.45rem;">'
        f'{lucide_svg} <span>{title}</span>'
        f'</div>'
        f'<div style="font-size: 0.88rem; color: #E2E8F0; margin-bottom: 0.35rem; line-height: 1.5;">'
        f'<strong>Insight:</strong> {observation}'
        f'</div>'
        f'<div style="font-size: 0.85rem; color: #94A3B8; margin-bottom: 0.35rem; line-height: 1.4;">'
        f'<strong>Business Implication:</strong> {implication}'
        f'</div>'
        f'<div style="font-size: 0.85rem; color: #34D399; margin-bottom: 0.2rem; line-height: 1.4;">'
        f'<strong>Strategic Recommendation:</strong> {recommendation}'
        f'</div>'
        f'{metrics_html}'
        f'</div>'
    )
    render_html_block(html)
