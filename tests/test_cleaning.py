"""
Tests for CleaningEngine.
"""

import pandas as pd
import numpy as np
from backend.cleaning_engine import CleaningEngine


def test_missing_value_imputation():
    df = pd.DataFrame({
        "num": [10.0, 20.0, np.nan, 40.0],
        "cat": ["A", "B", np.nan, "A"]
    })
    engine = CleaningEngine(df)

    # Impute numeric median
    log1 = engine.impute_missing_numerical("num", strategy="median")
    assert engine.cleaned_df["num"].isna().sum() == 0
    assert engine.cleaned_df["num"].iloc[2] == 20.0  # median of 10, 20, 40 is 20
    assert engine.original_df["num"].isna().sum() == 1  # original untouched!

    # Impute categorical mode
    log2 = engine.impute_missing_categorical("cat", strategy="mode")
    assert engine.cleaned_df["cat"].isna().sum() == 0
    assert engine.cleaned_df["cat"].iloc[2] == "A"
    assert len(engine.get_log()) == 2


def test_remove_duplicates():
    df = pd.DataFrame({"x": [1, 1, 2, 3]})
    engine = CleaningEngine(df)
    log = engine.remove_duplicates()
    assert len(engine.cleaned_df) == 3
    assert len(engine.original_df) == 4
    assert log["rows_affected"] == 1


def test_numeric_string_cleaning():
    df = pd.DataFrame({"price": ["$1,200.50", "$300.00", "50%"]})
    engine = CleaningEngine(df)
    engine.convert_to_numeric_clean("price")
    assert pd.api.types.is_numeric_dtype(engine.cleaned_df["price"])
    assert engine.cleaned_df["price"].iloc[0] == 1200.50
    assert engine.cleaned_df["price"].iloc[2] == 0.50


def test_outlier_capping():
    df = pd.DataFrame({"val": [10, 12, 11, 10, 9, 13, 1000]})
    engine = CleaningEngine(df)
    engine.handle_outliers("val", method="iqr", action="cap", threshold=1.5)
    assert engine.cleaned_df["val"].max() < 1000

