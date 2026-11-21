"""
Data Loader Engine for DataSense AI.
Handles robust loading of CSV, XLSX, and XLS datasets with sheet selection,
encoding detection fallback, and delimiter sniffing.
"""

import io
import os
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
from utils.validators import validate_file_extension, validate_file_size, validate_dataframe


class DataLoader:
    """Robust dataset loader for tabular business files."""

    @staticmethod
    def get_excel_sheet_names(file_source: Union[str, bytes, io.BytesIO]) -> List[str]:
        """Retrieve sheet names from an Excel file without loading entire data."""
        try:
            excel_file = pd.ExcelFile(file_source)
            return excel_file.sheet_names
        except Exception as e:
            return []

    @staticmethod
    def load_dataset(
        file_source: Union[str, bytes, io.BytesIO],
        filename: str,
        sheet_name: Optional[str] = None
    ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Load dataset from file path or bytes buffer.
        Returns: (dataframe, error_message)
        """
        # Validate extension
        is_valid_ext, ext_err = validate_file_extension(filename)
        if not is_valid_ext:
            return None, ext_err

        _, ext = os.path.splitext(filename.lower())

        try:
            if ext == ".csv":
                df = DataLoader._load_csv(file_source)
            elif ext in [".xlsx", ".xls"]:
                df = DataLoader._load_excel(file_source, sheet_name=sheet_name)
            else:
                return None, f"Unsupported file extension: {ext}"

            # Validate loaded dataframe
            is_valid_df, df_err = validate_dataframe(df)
            if not is_valid_df:
                return None, df_err

            # Standardize string column headers
            df.columns = [str(c).strip() for c in df.columns]

            return df, None

        except Exception as e:
            return None, f"Failed to parse file: {str(e)}"

    @staticmethod
    def _load_csv(file_source: Union[str, bytes, io.BytesIO]) -> pd.DataFrame:
        """Load CSV with automatic encoding and delimiter fallback."""
        encodings = ["utf-8", "latin-1", "iso-8859-1", "cp1252"]
        
        # If file_source is bytes, wrap in BytesIO
        if isinstance(file_source, bytes):
            buffer = io.BytesIO(file_source)
        elif isinstance(file_source, io.BytesIO):
            buffer = file_source
        elif isinstance(file_source, str) and os.path.exists(file_source):
            with open(file_source, "rb") as f:
                buffer = io.BytesIO(f.read())
        else:
            buffer = file_source

        last_error = None
        for enc in encodings:
            try:
                buffer.seek(0)
                # Let pandas sniff separator via sep=None, engine='python'
                df = pd.read_csv(buffer, encoding=enc, sep=None, engine="python")
                return df
            except Exception as e:
                last_error = e
                continue

        # If sniffing failed, try explicit commas
        for enc in encodings:
            try:
                buffer.seek(0)
                df = pd.read_csv(buffer, encoding=enc, sep=",")
                return df
            except Exception as e:
                last_error = e
                continue

        raise ValueError(f"Could not decode CSV file. Error: {last_error}")

    @staticmethod
    def _load_excel(
        file_source: Union[str, bytes, io.BytesIO],
        sheet_name: Optional[str] = None
    ) -> pd.DataFrame:
        """Load Excel worksheet."""
        if isinstance(file_source, bytes):
            buffer = io.BytesIO(file_source)
        elif isinstance(file_source, io.BytesIO):
            buffer = file_source
        elif isinstance(file_source, str) and os.path.exists(file_source):
            buffer = file_source
        else:
            buffer = file_source

        # If sheet_name is None, loads first sheet by default
        return pd.read_excel(buffer, sheet_name=sheet_name or 0)

