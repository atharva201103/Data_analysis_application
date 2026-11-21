"""
Automated Data Profiler Engine for DataSense AI.
Performs structural analysis, column-level statistics, data quality audit,
and calculates a transparent Dataset Health Score.
"""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from utils.helpers import format_bytes, format_number, format_percentage
from backend.column_detector import ColumnDetector


class DataProfiler:
    """Automated profiler providing structural insights and health diagnostics."""

    @staticmethod
    def profile_dataset(df: pd.DataFrame, column_roles: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive profile of the dataset.
        """
        total_rows = len(df)
        total_cols = len(df.columns)
        total_cells = total_rows * total_cols if total_rows > 0 else 1

        # Memory usage
        memory_bytes = int(df.memory_usage(deep=True).sum())
        memory_formatted = format_bytes(memory_bytes)

        # Duplicates
        duplicate_rows = int(df.duplicated().sum())
        duplicate_percentage = (duplicate_rows / total_rows * 100) if total_rows > 0 else 0.0

        # Type categorization
        numeric_cols = []
        categorical_cols = []
        datetime_cols = []
        boolean_cols = []
        identifier_cols = []
        constant_cols = []

        column_details = []
        total_missing_cells = 0

        for col in df.columns:
            series = df[col]
            dtype_str = str(series.dtype)
            null_count = int(series.isna().sum())
            total_missing_cells += null_count
            null_pct = (null_count / total_rows * 100) if total_rows > 0 else 0.0
            n_unique = int(series.nunique(dropna=True))

            # Role from detector or heuristic
            role = "General"
            if column_roles and col in column_roles:
                role = column_roles[col].get("role", "General")

            # Constant column
            is_constant = (n_unique <= 1 and total_rows > 1)
            if is_constant:
                constant_cols.append(col)

            # High cardinality / ID candidate
            is_id = (role == "Identifier") or (n_unique / total_rows > 0.90 and total_rows > 50 and not pd.api.types.is_numeric_dtype(series))
            if is_id and col not in identifier_cols:
                identifier_cols.append(col)

            # Categorize type
            if pd.api.types.is_datetime64_any_dtype(series):
                datetime_cols.append(col)
                col_type = "DateTime"
                min_val = str(series.min()) if null_count < total_rows else "N/A"
                max_val = str(series.max()) if null_count < total_rows else "N/A"
                mean_val = "N/A"
                median_val = "N/A"
            elif pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
                numeric_cols.append(col)
                col_type = "Numeric"
                valid_series = series.dropna()
                if len(valid_series) > 0:
                    min_val = format_number(float(valid_series.min()))
                    max_val = format_number(float(valid_series.max()))
                    mean_val = format_number(float(valid_series.mean()))
                    median_val = format_number(float(valid_series.median()))
                else:
                    min_val, max_val, mean_val, median_val = "N/A", "N/A", "N/A", "N/A"
            elif pd.api.types.is_bool_dtype(series):
                boolean_cols.append(col)
                col_type = "Boolean"
                min_val = str(series.min()) if null_count < total_rows else "N/A"
                max_val = str(series.max()) if null_count < total_rows else "N/A"
                mean_val = "N/A"
                median_val = "N/A"
            else:
                categorical_cols.append(col)
                col_type = "Categorical"
                valid_series = series.dropna()
                min_val = "N/A"
                max_val = "N/A"
                mean_val = "N/A"
                median_val = "N/A"

            column_details.append({
                "column_name": col,
                "data_type": dtype_str,
                "general_type": col_type,
                "inferred_role": role,
                "missing_count": null_count,
                "missing_percentage": round(null_pct, 2),
                "unique_count": n_unique,
                "min": min_val,
                "max": max_val,
                "mean": mean_val,
                "median": median_val,
                "is_constant": is_constant,
                "is_identifier": is_id
            })

        total_missing_percentage = (total_missing_cells / total_cells * 100) if total_cells > 0 else 0.0

        # Health score calculation
        health_score, score_breakdown, grade, grade_explanation = DataProfiler.calculate_health_score(
            total_rows=total_rows,
            total_cols=total_cols,
            missing_pct=total_missing_percentage,
            duplicate_pct=duplicate_percentage,
            constant_cols_count=len(constant_cols),
            column_details=column_details
        )

        return {
            "total_rows": total_rows,
            "total_columns": total_cols,
            "total_cells": total_cells,
            "memory_bytes": memory_bytes,
            "memory_formatted": memory_formatted,
            "duplicate_rows": duplicate_rows,
            "duplicate_percentage": round(duplicate_percentage, 2),
            "total_missing_cells": total_missing_cells,
            "total_missing_percentage": round(total_missing_percentage, 2),
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "datetime_columns": datetime_cols,
            "boolean_columns": boolean_cols,
            "identifier_columns": identifier_cols,
            "constant_columns": constant_cols,
            "column_details": column_details,
            "health_score": health_score,
            "health_grade": grade,
            "health_explanation": grade_explanation,
            "score_breakdown": score_breakdown
        }

    @staticmethod
    def calculate_health_score(
        total_rows: int,
        total_cols: int,
        missing_pct: float,
        duplicate_pct: float,
        constant_cols_count: int,
        column_details: List[Dict[str, Any]]
    ) -> Tuple[int, Dict[str, Any], str, str]:
        """
        Calculate transparent health score (0-100) based on weighted quality dimensions:
        1. Cell Completeness (35 pts max): penalizes missing values across cells.
        2. Row Uniqueness (25 pts max): penalizes duplicate records.
        3. Column Information Richness (20 pts max): penalizes constant or completely empty columns.
        4. High-Missing Columns Penalty (20 pts max): penalizes columns with >50% missing values.
        """
        if total_rows == 0 or total_cols == 0:
            return 0, {}, "Critical", "The dataset contains no usable rows or columns."

        # 1. Cell Completeness (35 pts)
        completeness_score = max(0.0, 35.0 * (1.0 - (missing_pct / 100.0)))

        # 2. Row Uniqueness (25 pts)
        uniqueness_score = max(0.0, 25.0 * (1.0 - (duplicate_pct / 100.0)))

        # 3. Column Information Richness (20 pts)
        constant_ratio = constant_cols_count / total_cols if total_cols > 0 else 0
        richness_score = max(0.0, 20.0 * (1.0 - constant_ratio))

        # 4. Severe column missingness (20 pts)
        severe_missing_cols = sum(1 for c in column_details if c["missing_percentage"] > 50.0)
        severe_ratio = severe_missing_cols / total_cols if total_cols > 0 else 0
        column_quality_score = max(0.0, 20.0 * (1.0 - severe_ratio))

        raw_score = completeness_score + uniqueness_score + richness_score + column_quality_score
        final_score = int(round(np.clip(raw_score, 0, 100)))

        breakdown = {
            "completeness_score": round(completeness_score, 1),
            "completeness_max": 35,
            "uniqueness_score": round(uniqueness_score, 1),
            "uniqueness_max": 25,
            "richness_score": round(richness_score, 1),
            "richness_max": 20,
            "column_quality_score": round(column_quality_score, 1),
            "column_quality_max": 20,
            "total_score": final_score
        }

        if final_score >= 90:
            grade = "A (Excellent)"
            explanation = (
                "Dataset is in exceptional condition with near-complete data, minimal duplicates, "
                "and robust informational value across all features."
            )
        elif final_score >= 75:
            grade = "B (Good)"
            explanation = (
                "Dataset is in healthy condition with minor data hygiene issues (e.g. isolated missing "
                "values or low duplicate count) that can be remediated with automated cleaning."
            )
        elif final_score >= 60:
            grade = "C (Fair)"
            explanation = (
                "Dataset contains moderate data quality concerns such as missing entries, duplicated rows, "
                "or uninformative columns. Cleaning is strongly recommended before modeling."
            )
        else:
            grade = "D (Poor / Needs Attention)"
            explanation = (
                "Significant data quality issues detected, including heavy missingness, high duplication, "
                "or constant features. Careful review and cleaning are required."
            )

        return final_score, breakdown, grade, explanation
