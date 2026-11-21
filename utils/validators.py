"""
Validation utilities for DataSense AI.
"""

from typing import Tuple, Optional
import os
import pandas as pd
from utils.constants import SUPPORTED_EXTENSIONS, MAX_UPLOAD_SIZE_BYTES


def validate_file_extension(filename: str) -> Tuple[bool, Optional[str]]:
    """Verify that file extension is supported."""
    if not filename:
        return False, "No filename provided."
    _, ext = os.path.splitext(filename.lower())
    if ext not in SUPPORTED_EXTENSIONS:
        return False, f"Unsupported file format '{ext}'. Please upload a CSV or Excel (.xlsx, .xls) file."
    return True, None


def validate_file_size(size_bytes: int, max_bytes: int = MAX_UPLOAD_SIZE_BYTES) -> Tuple[bool, Optional[str]]:
    """Verify that file size does not exceed the limit."""
    if size_bytes <= 0:
        return False, "File is empty (0 bytes)."
    if size_bytes > max_bytes:
        max_mb = max_bytes / (1024 * 1024)
        actual_mb = size_bytes / (1024 * 1024)
        return False, f"File size ({actual_mb:.1f} MB) exceeds maximum allowed limit of {max_mb:.0f} MB."
    return True, None


def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
    """Check whether a loaded dataframe is valid and non-empty."""
    if df is None:
        return False, "Dataframe could not be initialized."
    if not isinstance(df, pd.DataFrame):
        return False, "Provided object is not a valid Pandas DataFrame."
    if df.empty:
        return False, "The uploaded dataset contains zero rows."
    if len(df.columns) == 0:
        return False, "The uploaded dataset contains zero columns."
    return True, None

