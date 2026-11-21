"""
End-to-End Simulation Test for DataSense AI.
Validates the complete workflow across Retail Sales, HR Analytics, and E-Commerce datasets.
"""

import os
import pytest
import pandas as pd
from backend.data_loader import DataLoader
from backend.column_detector import ColumnDetector
from backend.data_profiler import DataProfiler
from backend.cleaning_engine import CleaningEngine
from backend.eda_engine import EDAEngine
from backend.visualization_engine import VisualizationEngine
from backend.kpi_engine import KPIEngine
from backend.business_rules import BusinessRuleEngine
from backend.insight_engine import InsightEngine
from backend.chat_engine import ChatEngine
from backend.report_generator import ReportGenerator


def test_retail_sales_complete_workflow():
    sample_file = "sample_datasets/retail_sales_sample.csv"
    assert os.path.exists(sample_file)

    # 1. Load Data
    df, err = DataLoader.load_dataset(sample_file, "retail_sales_sample.csv")
    assert err is None
    assert df is not None
    initial_rows = len(df)
    assert initial_rows > 200

    # 2. Detect Columns & Semantic Roles
    detector = ColumnDetector()
    roles = detector.detect_all(df)
    assert "Transaction_ID" in roles
    assert "Date" in roles
    assert "Category" in roles

    # 3. Data Profiling & Health Score
    profile = DataProfiler.profile_dataset(df, column_roles=roles)
    assert profile["total_rows"] == initial_rows
    assert profile["health_score"] > 0
    assert profile["duplicate_rows"] > 0
    assert profile["total_missing_cells"] > 0

    # 4. Data Cleaning
    engine = CleaningEngine(df)

    # Check that formatted strings like discount '15%' can be cleaned
    engine.convert_to_numeric_clean("Discount")
    assert pd.api.types.is_numeric_dtype(engine.cleaned_df["Discount"])

    # Remove duplicates
    dup_log = engine.remove_duplicates()
    assert dup_log["rows_affected"] > 0
    assert len(engine.cleaned_df) < initial_rows
    assert len(engine.original_df) == initial_rows  # Original strictly untouched!

    # Impute missing Profit
    engine.impute_missing_numerical("Profit", strategy="median")
    assert engine.cleaned_df["Profit"].isna().sum() == 0

    # Impute missing Payment_Method
    engine.impute_missing_categorical("Payment_Method", strategy="mode")
    assert engine.cleaned_df["Payment_Method"].isna().sum() == 0

    # Standardize category text
    engine.standardize_text("Category", case="title")

    # Audit summary
    audit_summary = engine.get_before_after_summary()
    assert audit_summary["rows_removed"] > 0
    assert audit_summary["missing_cells_resolved"] > 0

    cleaned_df = engine.get_cleaned_df()

    # 5. EDA Calculations
    eda_stats = EDAEngine.get_numerical_summary(cleaned_df)
    assert not eda_stats.empty
    assert "Total_Revenue" in eda_stats["Column"].values

    corr_df, top_pairs = EDAEngine.get_correlation_matrix(cleaned_df)
    assert not corr_df.empty
    assert len(top_pairs) > 0

    # 6. KPI Engine
    kpis = KPIEngine.compute_kpis(cleaned_df, roles, domain="retail_sales")
    assert len(kpis) >= 3
    kpi_titles = [k["title"] for k in kpis]
    assert any("Revenue" in t for t in kpi_titles)

    # 7. Business Rules & Insights
    findings = BusinessRuleEngine.evaluate_rules(cleaned_df, roles, domain="retail_sales")
    assert len(findings) > 0

    insight_engine = InsightEngine()
    insights = insight_engine.generate_all_insights(cleaned_df, roles, domain="retail_sales")
    assert len(insights) > 0
    for ins in insights:
        assert "title" in ins
        assert "observation" in ins
        assert "implication" in ins
        assert "recommendation" in ins

    # 8. Visualizations
    charts = VisualizationEngine.generate_recommended_charts(cleaned_df, roles)
    assert len(charts) > 0

    # 9. Chat Engine Queries
    chat = ChatEngine(cleaned_df, roles)
    res1 = chat.process_query("Which category generated the highest revenue?")
    assert "Electronics" in res1["answer"]
    assert res1["figure"] is not None

    res2 = chat.process_query("What is the average Total_Revenue?")
    assert "$" in res2["answer"] or "average" in res2["answer"].lower()

    res3 = chat.process_query("Show monthly revenue trends")
    assert res3["figure"] is not None

    # 10. PDF & Excel Report Generation
    pdf_bytes = ReportGenerator.generate_pdf_report(
        df=cleaned_df,
        profile_data=profile,
        kpis=kpis,
        insights=insights,
        cleaning_log=engine.get_log(),
        dataset_name="Retail Sales Test"
    )
    assert pdf_bytes.startswith(b"%PDF")

    excel_bytes = ReportGenerator.generate_excel_workbook(
        cleaned_df=cleaned_df,
        profile_data=profile,
        kpis=kpis,
        cleaning_log=engine.get_log(),
        eda_summary=eda_stats
    )
    assert excel_bytes.startswith(b"PK")


def test_hr_analytics_workflow():
    sample_file = "sample_datasets/hr_analytics_sample.csv"
    assert os.path.exists(sample_file)

    df, err = DataLoader.load_dataset(sample_file, "hr_analytics_sample.csv")
    assert err is None

    detector = ColumnDetector()
    roles = detector.detect_all(df)

    domain = KPIEngine.identify_domain(df, roles)
    assert domain == "hr_analytics"

    kpis = KPIEngine.compute_kpis(df, roles, domain=domain)
    kpi_ids = [k["id"] for k in kpis]
    assert "hr_headcount" in kpi_ids
    assert "hr_avg_salary" in kpi_ids
    assert "hr_attrition_rate" in kpi_ids

    findings = BusinessRuleEngine.evaluate_rules(df, roles, domain=domain)
    assert any(f["rule_id"] == "hr_overtime_attrition_risk" for f in findings)


def test_ecommerce_orders_workflow():
    sample_file = "sample_datasets/ecommerce_orders_sample.csv"
    assert os.path.exists(sample_file)

    df, err = DataLoader.load_dataset(sample_file, "ecommerce_orders_sample.csv")
    assert err is None

    detector = ColumnDetector()
    roles = detector.detect_all(df)

    domain = KPIEngine.identify_domain(df, roles)
    assert domain == "ecommerce"

    kpis = KPIEngine.compute_kpis(df, roles, domain=domain)
    assert len(kpis) > 0

