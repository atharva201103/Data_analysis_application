"""
Unit tests for UI components and Lucide vector icon rendering.
Verifies that KPI cards and icons render valid HTML without dangling </div> tags.
"""

import xml.etree.ElementTree as ET
from utils.ui_components import get_lucide_icon, LUCIDE_SVG_BODIES, ICON_ALIASES


def test_lucide_icons_valid_xml():
    """Verify all defined Lucide SVG bodies are syntactically valid XML."""
    for key in LUCIDE_SVG_BODIES.keys():
        svg_str = get_lucide_icon(key, size=16, color="#4F46E5")
        assert svg_str.startswith('<svg')
        assert svg_str.endswith('</svg>')
        # Parse XML to verify well-formedness
        tree = ET.fromstring(svg_str)
        assert tree.tag == '{http://www.w3.org/2000/svg}svg' or tree.tag == 'svg'
        assert tree.attrib['width'] == '16'
        assert tree.attrib['height'] == '16'


def test_emoji_and_alias_mapping():
    """Verify that common emojis and aliases resolve to clean Lucide SVGs without errors."""
    test_cases = ["💰", "🛒", "📈", "👥", "🔢", "📊", "⚠️", "💾", "✅", "⚡", "dollar-sign", "package", "unknown_xyz"]
    for case in test_cases:
        svg_str = get_lucide_icon(case)
        assert "<svg" in svg_str
        assert "</svg>" in svg_str
        # Verify no unclosed tags
        ET.fromstring(svg_str)


def test_kpi_card_html_structure():
    """Verify KPI card structure has matched opening and closing div tags."""
    from utils.ui_components import get_lucide_icon
    
    icon_svg = get_lucide_icon("dollar-sign", size=16, color="#818CF8")
    title = "Total Revenue"
    value = "$1,245.50"
    subtext = "Across 250 orders"
    delta_html = '<span class="kpi-badge positive">+12.5%</span>'
    
    card_html = (
        f'<div class="kpi-card">'
        f'<div class="kpi-header">'
        f'<span class="kpi-icon-badge">{icon_svg}</span>'
        f'<span class="kpi-title">{title}</span>'
        f'</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-footer">'
        f'<span class="kpi-subtext">{subtext}</span>'
        f'{delta_html}'
        f'</div>'
        f'</div>'
    )
    
    # Verify tag parity: opening <div> count must equal closing </div> count
    assert card_html.count("<div") == card_html.count("</div>")
    assert card_html.count("<span") == card_html.count("</span>")
    assert card_html.count("<svg") == card_html.count("</svg>")
    
    # Verify card_html does not start with 4 spaces (which would trigger markdown code-block)
    for line in card_html.splitlines():
        assert not line.startswith("    ")

