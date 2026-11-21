"""
Tests for ReportGenerator.
"""

import pandas as pd
from backend.report_generator import ReportGenerator
from backend.data_profiler import DataProfiler


def test_pdf_and_excel_generation():
    df = pd.DataFrame({
        "Category": ["A", "B", "C"],
        "Sales": [100.0, 200.0, 300.0]
    })
    profile = DataProfiler.profile_dataset(df)
    kpis = [{"title": "Total Sales", "value": "$600.00", "category": "Sales", "subtext": "Sum"}]
    insights = [{
        "title": "Category Dominance",
        "observation": "Category C leads.",
        "implication": "Core revenue driver.",
        "recommendation": "Protect market share."
    }]
    clean_log = [{"timestamp": "2024-01-01", "operation": "Test", "rows_affected": 0, "details": "Init"}]

    # PDF generation
    pdf_bytes = ReportGenerator.generate_pdf_report(
        df=df,
        profile_data=profile,
        kpis=kpis,
        insights=insights,
        cleaning_log=clean_log
    )
    assert pdf_bytes is not None
    assert len(pdf_bytes) > 500  # Non-trivial PDF output
    assert pdf_bytes.startswith(b"%PDF")

    # Excel generation
    excel_bytes = ReportGenerator.generate_excel_workbook(
        cleaned_df=df,
        profile_data=profile,
        kpis=kpis,
        cleaning_log=clean_log
    )
    assert excel_bytes is not None
    assert len(excel_bytes) > 500
    assert excel_bytes.startswith(b"PK")  # Zip/OpenPyXL signature

