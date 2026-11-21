"""
Tests for BusinessRuleEngine & InsightEngine.
"""

import pandas as pd
from backend.column_detector import ColumnDetector
from backend.business_rules import BusinessRuleEngine
from backend.insight_engine import InsightEngine
from backend.kpi_engine import KPIEngine


def test_business_rules_and_kpis():
    df = pd.DataFrame({
        "Category": ["Electronics", "Electronics", "Fashion", "Office Supplies", "Office Supplies"],
        "Total_Revenue": [1000.0, 1500.0, 300.0, 200.0, 150.0],
        "Profit": [300.0, 450.0, -50.0, 40.0, 30.0],
        "Region": ["North", "North", "South", "East", "West"]
    })
    detector = ColumnDetector()
    roles = detector.detect_all(df)

    kpis = KPIEngine.compute_kpis(df, roles)
    assert len(kpis) > 0
    kpi_ids = [k["id"] for k in kpis]
    assert "total_revenue" in kpi_ids
    assert "total_profit" in kpi_ids

    findings = BusinessRuleEngine.evaluate_rules(df, roles)
    assert len(findings) > 0

    engine = InsightEngine()
    insights = engine.generate_all_insights(df, roles)
    assert len(insights) > 0
    first = insights[0]
    assert "observation" in first
    assert "implication" in first
    assert "recommendation" in first

