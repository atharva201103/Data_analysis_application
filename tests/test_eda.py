"""
Tests for EDAEngine.
"""

import pandas as pd
import numpy as np
from backend.eda_engine import EDAEngine


def test_numerical_summary():
    df = pd.DataFrame({
        "sales": [100, 200, 300, 400, 500],
        "profit": [20, 40, 60, 80, 100]
    })
    summary = EDAEngine.get_numerical_summary(df)
    assert not summary.empty
    assert "Skewness" in summary.columns
    assert "Kurtosis" in summary.columns
    assert summary.loc[summary["Column"] == "sales", "Mean"].iloc[0] == 300.0


def test_correlation_matrix():
    df = pd.DataFrame({
        "x": [1, 2, 3, 4, 5],
        "y": [2, 4, 6, 8, 10]
    })
    corr_df, pairs = EDAEngine.get_correlation_matrix(df)
    assert not corr_df.empty
    assert len(pairs) == 1
    assert pairs[0]["correlation"] == 1.0

