"""
Automated Data Cleaning Engine for DataSense AI.
Provides modular, non-destructive cleaning transformations with audit logging,
before-and-after tracking, and preview mechanisms.
"""

from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np


class CleaningEngine:
    """Non-destructive data cleaning engine with audit logging."""

    def __init__(self, df: pd.DataFrame):
        # Store an immutable copy of original
        self.original_df = df.copy(deep=True)
        # Working cleaned copy
        self.cleaned_df = df.copy(deep=True)
        # Operation log
        self.operation_log: List[Dict[str, Any]] = []

    def get_cleaned_df(self) -> pd.DataFrame:
        """Return the current cleaned dataframe."""
        return self.cleaned_df

    def get_original_df(self) -> pd.DataFrame:
        """Return the untouched original dataframe."""
        return self.original_df

    def get_log(self) -> List[Dict[str, Any]]:
        """Return audit trail of applied transformations."""
        return self.operation_log

    def reset_to_original(self):
        """Revert all cleaning operations back to original dataset."""
        self.cleaned_df = self.original_df.copy(deep=True)
        self.operation_log = [{
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operation": "Reset",
            "details": "Reverted all transformations back to original uploaded state.",
            "rows_affected": len(self.cleaned_df),
            "cols_affected": len(self.cleaned_df.columns)
        }]

    # -------------------------------------------------------------
    # A. Missing Value Handlers
    # -------------------------------------------------------------
    def drop_missing_rows(self, subset_cols: Optional[List[str]] = None, how: str = "any") -> Dict[str, Any]:
        """Drop rows containing null values."""
        before_rows = len(self.cleaned_df)
        self.cleaned_df = self.cleaned_df.dropna(subset=subset_cols, how=how)
        after_rows = len(self.cleaned_df)
        dropped_count = before_rows - after_rows

        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operation": "Drop Missing Rows",
            "details": f"Removed {dropped_count} rows with missing values (how='{how}', columns={subset_cols or 'all'}).",
            "rows_affected": dropped_count,
            "cols_affected": 0
        }
        self.operation_log.append(log_entry)
        return log_entry

    def drop_missing_columns(self, threshold_pct: float = 50.0) -> Dict[str, Any]:
        """Drop columns exceeding a given missing percentage threshold."""
        before_cols = list(self.cleaned_df.columns)
        cols_to_drop = [
            c for c in self.cleaned_df.columns
            if (self.cleaned_df[c].isna().sum() / len(self.cleaned_df) * 100) >= threshold_pct
        ]
        if cols_to_drop:
            self.cleaned_df = self.cleaned_df.drop(columns=cols_to_drop)

        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operation": "Drop High-Missing Columns",
            "details": f"Dropped {len(cols_to_drop)} columns exceeding {threshold_pct}% missingness: {cols_to_drop}",
            "rows_affected": 0,
            "cols_affected": len(cols_to_drop)
        }
        self.operation_log.append(log_entry)
        return log_entry

    def impute_missing_numerical(self, column: str, strategy: str = "median") -> Dict[str, Any]:
        """Impute missing numeric values using mean, median, zero, or forward/backward fill."""
        if column not in self.cleaned_df.columns:
            raise ValueError(f"Column '{column}' does not exist.")

        null_count = int(self.cleaned_df[column].isna().sum())
        if null_count == 0:
            return {"details": f"No missing values in column '{column}'."}

        series = self.cleaned_df[column]
        if strategy == "median":
            fill_val = series.median()
            self.cleaned_df[column] = series.fillna(fill_val)
        elif strategy == "mean":
            fill_val = series.mean()
            self.cleaned_df[column] = series.fillna(fill_val)
        elif strategy == "zero":
            fill_val = 0
            self.cleaned_df[column] = series.fillna(0)
        elif strategy == "ffill":
            fill_val = "forward fill"
            self.cleaned_df[column] = series.ffill()
        elif strategy == "bfill":
            fill_val = "backward fill"
            self.cleaned_df[column] = series.bfill()
        else:
            raise ValueError(f"Unknown imputation strategy: {strategy}")

        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operation": f"Impute Numerical ({strategy.capitalize()})",
            "details": f"Imputed {null_count} missing values in '{column}' using strategy '{strategy}' (value: {fill_val}).",
            "rows_affected": null_count,
            "cols_affected": 1
        }
        self.operation_log.append(log_entry)
        return log_entry

    def impute_missing_categorical(self, column: str, strategy: str = "mode", custom_val: Optional[str] = None) -> Dict[str, Any]:
        """Impute missing categorical values using mode, 'Unknown', or custom string."""
        if column not in self.cleaned_df.columns:
            raise ValueError(f"Column '{column}' does not exist.")

        null_count = int(self.cleaned_df[column].isna().sum())
        if null_count == 0:
            return {"details": f"No missing values in column '{column}'."}

        if strategy == "mode":
            mode_series = self.cleaned_df[column].mode()
            fill_val = mode_series.iloc[0] if len(mode_series) > 0 else "Unknown"
        elif strategy == "custom":
            fill_val = custom_val if custom_val is not None else "Unknown"
        else:
            fill_val = "Missing"

        self.cleaned_df[column] = self.cleaned_df[column].fillna(fill_val)

        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operation": f"Impute Categorical ({strategy})",
            "details": f"Imputed {null_count} missing values in '{column}' with '{fill_val}'.",
            "rows_affected": null_count,
            "cols_affected": 1
        }
        self.operation_log.append(log_entry)
        return log_entry

    # -------------------------------------------------------------
    # B. Duplicate Records Handlers
    # -------------------------------------------------------------
    def remove_duplicates(self, subset_cols: Optional[List[str]] = None, keep: str = "first") -> Dict[str, Any]:
        """Remove duplicate records."""
        before_rows = len(self.cleaned_df)
        self.cleaned_df = self.cleaned_df.drop_duplicates(subset=subset_cols, keep=keep)
        after_rows = len(self.cleaned_df)
        removed_count = before_rows - after_rows

        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operation": "Remove Duplicates",
            "details": f"Removed {removed_count} duplicate rows (keep='{keep}', subset={subset_cols or 'all'}).",
            "rows_affected": removed_count,
            "cols_affected": 0
        }
        self.operation_log.append(log_entry)
        return log_entry

    # -------------------------------------------------------------
    # C. Data Type Conversions
    # -------------------------------------------------------------
    def convert_to_numeric_clean(self, column: str) -> Dict[str, Any]:
        """
        Clean and convert numeric strings containing $, commas, %, or currency symbols to numeric float.
        """
        if column not in self.cleaned_df.columns:
            raise ValueError(f"Column '{column}' does not exist.")

        series = self.cleaned_df[column].astype(str)
        # Strip $, commas, spaces, %
        cleaned_series = (
            series.str.replace("$", "", regex=False)
                  .str.replace("€", "", regex=False)
                  .str.replace("£", "", regex=False)
                  .str.replace("₹", "", regex=False)
                  .str.replace(",", "", regex=False)
                  .str.replace(" ", "", regex=False)
        )

        # Handle percentage (e.g. 15% -> 0.15)
        is_pct = cleaned_series.str.endswith("%")
        cleaned_series = cleaned_series.str.replace("%", "", regex=False)

        numeric_vals = pd.to_numeric(cleaned_series, errors="coerce")
        # If percentage, divide by 100
        numeric_vals = np.where(is_pct, numeric_vals / 100.0, numeric_vals)

        converted_series = pd.Series(numeric_vals, index=self.cleaned_df.index)
        self.cleaned_df[column] = converted_series

        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operation": "Convert to Numeric",
            "details": f"Standardized formatted strings in column '{column}' to clean numeric values.",
            "rows_affected": len(self.cleaned_df),
            "cols_affected": 1
        }
        self.operation_log.append(log_entry)
        return log_entry

    def convert_to_datetime(self, column: str, date_format: Optional[str] = None) -> Dict[str, Any]:
        """Convert a date/time string column into standardized datetime."""
        if column not in self.cleaned_df.columns:
            raise ValueError(f"Column '{column}' does not exist.")

        try:
            if date_format:
                self.cleaned_df[column] = pd.to_datetime(self.cleaned_df[column], format=date_format, errors="coerce")
            else:
                self.cleaned_df[column] = pd.to_datetime(self.cleaned_df[column], errors="coerce")
        except Exception as e:
            raise ValueError(f"Datetime conversion failed: {str(e)}")

        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operation": "Convert to Datetime",
            "details": f"Parsed column '{column}' to datetime objects (format: {date_format or 'auto'}).",
            "rows_affected": len(self.cleaned_df),
            "cols_affected": 1
        }
        self.operation_log.append(log_entry)
        return log_entry

    # -------------------------------------------------------------
    # D. Outlier Detection & Treatment
    # -------------------------------------------------------------
    def detect_outliers_iqr(self, column: str, multiplier: float = 1.5) -> Tuple[pd.Series, float, float, int]:
        """
        Detect outliers using IQR method.
        Returns: (boolean mask of outliers, lower_bound, upper_bound, count)
        """
        series = self.cleaned_df[column].dropna()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (multiplier * iqr)
        upper_bound = q3 + (multiplier * iqr)

        outlier_mask = (self.cleaned_df[column] < lower_bound) | (self.cleaned_df[column] > upper_bound)
        count = int(outlier_mask.sum())
        return outlier_mask, lower_bound, upper_bound, count

    def detect_outliers_zscore(self, column: str, threshold: float = 3.0) -> Tuple[pd.Series, float, float, int]:
        """
        Detect outliers using Z-score method.
        Returns: (boolean mask of outliers, lower_bound, upper_bound, count)
        """
        series = self.cleaned_df[column].dropna()
        mean = series.mean()
        std = series.std()
        if std == 0 or pd.isna(std):
            mask = pd.Series(False, index=self.cleaned_df.index)
            return mask, mean, mean, 0

        lower_bound = mean - (threshold * std)
        upper_bound = mean + (threshold * std)
        outlier_mask = (self.cleaned_df[column] < lower_bound) | (self.cleaned_df[column] > upper_bound)
        count = int(outlier_mask.sum())
        return outlier_mask, lower_bound, upper_bound, count

    def handle_outliers(self, column: str, method: str = "iqr", action: str = "cap", threshold: float = 1.5) -> Dict[str, Any]:
        """
        Handle outliers either by capping (Winsorizing) or dropping.
        """
        if method == "iqr":
            mask, lower_bound, upper_bound, count = self.detect_outliers_iqr(column, multiplier=threshold)
        else:
            mask, lower_bound, upper_bound, count = self.detect_outliers_zscore(column, threshold=threshold)

        if count == 0:
            return {"details": f"No outliers found in '{column}' using {method.upper()}."}

        if action == "cap":
            self.cleaned_df[column] = self.cleaned_df[column].clip(lower=lower_bound, upper=upper_bound)
            details = f"Capped {count} outliers in '{column}' to range [{lower_bound:.2f}, {upper_bound:.2f}]."
            affected_rows = count
        elif action == "drop":
            before = len(self.cleaned_df)
            self.cleaned_df = self.cleaned_df[~mask]
            affected_rows = before - len(self.cleaned_df)
            details = f"Dropped {affected_rows} outlier rows from '{column}' using {method.upper()}."
        else:
            raise ValueError(f"Unknown outlier action: {action}")

        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operation": f"Handle Outliers ({method.upper()}-{action})",
            "details": details,
            "rows_affected": affected_rows,
            "cols_affected": 1
        }
        self.operation_log.append(log_entry)
        return log_entry

    # -------------------------------------------------------------
    # E. Inconsistent Data & Text Standardization
    # -------------------------------------------------------------
    def standardize_text(self, column: str, case: str = "title", strip_whitespace: bool = True) -> Dict[str, Any]:
        """Standardize text column by trimming whitespace and normalizing casing."""
        if column not in self.cleaned_df.columns:
            raise ValueError(f"Column '{column}' does not exist.")

        series = self.cleaned_df[column].astype(str)
        if strip_whitespace:
            series = series.str.strip()

        if case == "lower":
            series = series.str.lower()
        elif case == "upper":
            series = series.str.upper()
        elif case == "title":
            series = series.str.title()

        self.cleaned_df[column] = series

        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operation": f"Standardize Text ({case})",
            "details": f"Applied {case} case formatting and whitespace trimming to column '{column}'.",
            "rows_affected": len(self.cleaned_df),
            "cols_affected": 1
        }
        self.operation_log.append(log_entry)
        return log_entry

    # -------------------------------------------------------------
    # F. Column Operations (Rename, Drop)
    # -------------------------------------------------------------
    def rename_column(self, old_name: str, new_name: str) -> Dict[str, Any]:
        """Rename a column in the cleaned dataset."""
        if old_name not in self.cleaned_df.columns:
            raise ValueError(f"Column '{old_name}' does not exist.")

        clean_new = new_name.strip()
        self.cleaned_df = self.cleaned_df.rename(columns={old_name: clean_new})

        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operation": "Rename Column",
            "details": f"Renamed column '{old_name}' to '{clean_new}'.",
            "rows_affected": 0,
            "cols_affected": 1
        }
        self.operation_log.append(log_entry)
        return log_entry

    def drop_columns(self, columns: List[str]) -> Dict[str, Any]:
        """Drop selected columns from the cleaned dataset."""
        existing = [c for c in columns if c in self.cleaned_df.columns]
        if existing:
            self.cleaned_df = self.cleaned_df.drop(columns=existing)

        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operation": "Drop Columns",
            "details": f"Dropped {len(existing)} columns: {existing}",
            "rows_affected": 0,
            "cols_affected": len(existing)
        }
        self.operation_log.append(log_entry)
        return log_entry

    # -------------------------------------------------------------
    # Cleaning Recommendations & Audit Summary
    # -------------------------------------------------------------
    def get_cleaning_recommendations(self) -> List[Dict[str, Any]]:
        """Automatically analyze dataset and generate prioritized cleaning recommendations."""
        recommendations = []
        df = self.cleaned_df

        # 1. Duplicates
        dup_count = int(df.duplicated().sum())
        if dup_count > 0:
            recommendations.append({
                "type": "duplicates",
                "priority": "High" if dup_count > 10 else "Medium",
                "action": "Remove Duplicates",
                "description": f"Found {dup_count} exact duplicate rows ({dup_count/len(df)*100:.1f}% of data).",
                "suggested_fix": "Execute 'Remove Duplicates' keeping the first occurrence."
            })

        # 2. Missing values
        for col in df.columns:
            null_count = int(df[col].isna().sum())
            if null_count > 0:
                pct = (null_count / len(df)) * 100
                if pct > 60:
                    recommendations.append({
                        "type": "missing_high",
                        "priority": "High",
                        "column": col,
                        "action": "Drop Column",
                        "description": f"Column '{col}' has {null_count} missing values ({pct:.1f}%).",
                        "suggested_fix": f"Consider dropping '{col}' as over half the entries are missing."
                    })
                elif pd.api.types.is_numeric_dtype(df[col]):
                    recommendations.append({
                        "type": "missing_numeric",
                        "priority": "Medium",
                        "column": col,
                        "action": "Impute Median",
                        "description": f"Numeric column '{col}' has {null_count} missing values ({pct:.1f}%).",
                        "suggested_fix": f"Impute missing values using the median value ({df[col].median():.2f})."
                    })
                else:
                    recommendations.append({
                        "type": "missing_categorical",
                        "priority": "Medium",
                        "column": col,
                        "action": "Impute Mode / 'Unknown'",
                        "description": f"Categorical column '{col}' has {null_count} missing values ({pct:.1f}%).",
                        "suggested_fix": "Impute using the most frequent category or label as 'Unknown'."
                    })

        # 3. String numeric detection (contains currency signs or percentages)
        for col in df.select_dtypes(include=["object"]).columns:
            sample_vals = df[col].dropna().astype(str).head(20)
            has_currency = sample_vals.str.contains(r"[\$,€,£,₹]", regex=True).any()
            has_pct = sample_vals.str.contains(r"%", regex=True).any()
            if has_currency or has_pct:
                recommendations.append({
                    "type": "type_mismatch",
                    "priority": "High",
                    "column": col,
                    "action": "Convert to Numeric",
                    "description": f"Column '{col}' appears to contain formatted numeric values with currency or % symbols.",
                    "suggested_fix": "Strip non-numeric symbols and cast to float for analytical computations."
                })

        # 4. Outliers
        for col in df.select_dtypes(include=[np.number]).columns:
            _, _, _, count = self.detect_outliers_iqr(col)
            if count > 0 and count / len(df) < 0.10: # Less than 10% are extreme
                recommendations.append({
                    "type": "outlier",
                    "priority": "Low",
                    "column": col,
                    "action": "Review Outliers",
                    "description": f"Detected {count} statistical outliers in numeric column '{col}'.",
                    "suggested_fix": "Review distribution and optionally cap values using the IQR method."
                })

        return recommendations

    def get_before_after_summary(self) -> Dict[str, Any]:
        """Compute before-and-after audit statistics comparing original and cleaned."""
        orig = self.original_df
        clean = self.cleaned_df

        orig_missing = int(orig.isna().sum().sum())
        clean_missing = int(clean.isna().sum().sum())

        orig_dups = int(orig.duplicated().sum())
        clean_dups = int(clean.duplicated().sum())

        return {
            "original_rows": len(orig),
            "cleaned_rows": len(clean),
            "rows_removed": len(orig) - len(clean),
            "original_cols": len(orig.columns),
            "cleaned_cols": len(clean.columns),
            "cols_removed": len(orig.columns) - len(clean.columns),
            "original_missing_cells": orig_missing,
            "cleaned_missing_cells": clean_missing,
            "missing_cells_resolved": orig_missing - clean_missing,
            "original_duplicates": orig_dups,
            "cleaned_duplicates": clean_dups,
            "duplicates_removed": orig_dups - clean_dups,
            "total_operations_applied": len(self.operation_log)
        }

