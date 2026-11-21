"""
Preprocessing and Feature Engineering Utilities for DataSense AI.
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np


class Preprocessor:
    """Prepares and enriches tabular datasets."""

    @staticmethod
    def extract_datetime_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        """Extract Year, Month, Quarter, Day of Week from date column."""
        new_df = df.copy()
        if date_col in new_df.columns:
            dt_series = pd.to_datetime(new_df[date_col], errors="coerce")
            new_df[f"{date_col}_Year"] = dt_series.dt.year
            new_df[f"{date_col}_Month"] = dt_series.dt.month
            new_df[f"{date_col}_Quarter"] = dt_series.dt.quarter
            new_df[f"{date_col}_DayOfWeek"] = dt_series.dt.day_name()
        return new_df

    @staticmethod
    def get_column_type_mapping(df: pd.DataFrame) -> Dict[str, str]:
        """Map each column to high-level type: numeric, datetime, categorical, boolean."""
        types = {}
        for col in df.columns:
            s = df[col]
            if pd.api.types.is_datetime64_any_dtype(s):
                types[col] = "datetime"
            elif pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
                types[col] = "numeric"
            elif pd.api.types.is_bool_dtype(s):
                types[col] = "boolean"
            else:
                types[col] = "categorical"
        return types

