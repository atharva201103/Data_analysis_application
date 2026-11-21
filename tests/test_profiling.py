"""
Tests for DataProfiler.
"""

import pandas as pd
import numpy as np
from backend.data_profiler import DataProfiler


def test_profiler_statistics():
    df = pd.DataFrame({
        "A": [1, 2, np.nan, 4, 5],
        "B": ["cat", "dog", "dog", "cat", "bird"],
        "C": [10.0, 10.0, 10.0, 10.0, 10.0]  # constant column
    })
    profile = DataProfiler.profile_dataset(df)

    assert profile["total_rows"] == 5
    assert profile["total_columns"] == 3
    assert profile["total_missing_cells"] == 1
    assert "C" in profile["constant_columns"]
    assert profile["health_score"] > 0
    assert "health_grade" in profile
    assert len(profile["column_details"]) == 3

