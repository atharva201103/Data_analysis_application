"""
Tests for DataLoader.
"""

import os
import pytest
import pandas as pd
from backend.data_loader import DataLoader


def test_load_csv_sample():
    sample_path = "sample_datasets/retail_sales_sample.csv"
    assert os.path.exists(sample_path)
    df, err = DataLoader.load_dataset(sample_path, "retail_sales_sample.csv")
    assert err is None
    assert df is not None
    assert len(df) > 0
    assert "Transaction_ID" in df.columns


def test_load_excel_multisheet():
    excel_path = "sample_datasets/business_analytics_multisheet.xlsx"
    assert os.path.exists(excel_path)
    sheets = DataLoader.get_excel_sheet_names(excel_path)
    assert "Retail_Sales" in sheets
    assert "HR_Workforce" in sheets

    df_hr, err = DataLoader.load_dataset(excel_path, "business_analytics_multisheet.xlsx", sheet_name="HR_Workforce")
    assert err is None
    assert df_hr is not None
    assert "Employee_ID" in df_hr.columns


def test_invalid_extension():
    df, err = DataLoader.load_dataset("dummy.txt", "dummy.txt")
    assert df is None
    assert "Unsupported file format" in err

