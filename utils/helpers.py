"""
Helper utility functions for DataSense AI.
"""

from typing import Any, Union
import pandas as pd
import numpy as np


def format_currency(value: Union[int, float, None], prefix: str = "$") -> str:
    """Format numeric value into readable currency string with K/M/B abbreviations."""
    if value is None or pd.isna(value):
        return "N/A"
    try:
        val = float(value)
        sign = "-" if val < 0 else ""
        abs_val = abs(val)

        if abs_val >= 1_000_000_000:
            return f"{sign}{prefix}{abs_val / 1_000_000_000:,.2f}B"
        elif abs_val >= 1_000_000:
            return f"{sign}{prefix}{abs_val / 1_000_000:,.2f}M"
        elif abs_val >= 1_000:
            return f"{sign}{prefix}{abs_val / 1_000:,.2f}K"
        else:
            return f"{sign}{prefix}{abs_val:,.2f}"
    except (ValueError, TypeError):
        return str(value)


def format_number(value: Union[int, float, None], decimals: int = 2) -> str:
    """Format generic number into readable humanized string."""
    if value is None or pd.isna(value):
        return "N/A"
    try:
        val = float(value)
        sign = "-" if val < 0 else ""
        abs_val = abs(val)

        if abs_val >= 1_000_000_000:
            return f"{sign}{abs_val / 1_000_000_000:,.2f}B"
        elif abs_val >= 1_000_000:
            return f"{sign}{abs_val / 1_000_000:,.2f}M"
        elif abs_val >= 10_000:
            return f"{sign}{abs_val / 1_000:,.1f}K"
        elif abs_val == int(abs_val):
            return f"{sign}{int(abs_val):,}"
        else:
            return f"{sign}{abs_val:,.{decimals}f}"
    except (ValueError, TypeError):
        return str(value)


def format_percentage(value: Union[int, float, None], decimals: int = 1) -> str:
    """Format decimal or float into percentage string."""
    if value is None or pd.isna(value):
        return "N/A"
    try:
        val = float(value)
        # If value is already in 0-100 scale or 0-1 scale
        return f"{val:.{decimals}f}%"
    except (ValueError, TypeError):
        return str(value)


def format_bytes(num_bytes: int) -> str:
    """Convert bytes into human readable format (KB, MB, GB)."""
    if num_bytes is None or num_bytes < 0:
        return "0 B"
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:3.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"


def sanitize_column_name(col: Any) -> str:
    """Clean and standardize a column name."""
    if col is None:
        return "unnamed_column"
    return str(col).strip()


def safe_json_serialize(obj: Any) -> Any:
    """Convert numpy / pandas types to native JSON-serializable Python types."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return None if np.isnan(obj) else float(obj)
    elif isinstance(obj, (np.ndarray, pd.Series)):
        return [safe_json_serialize(x) for x in obj]
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    elif pd.isna(obj):
        return None
    elif isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: safe_json_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [safe_json_serialize(x) for x in obj]
    return obj

