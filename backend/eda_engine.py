"""
Exploratory Data Analysis (EDA) Engine for DataSense AI.
Computes comprehensive descriptive statistics, categorical distributions,
correlation matrices, and detects significant bivariate relationships.
"""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from scipy import stats
from utils.helpers import format_number, safe_json_serialize


class EDAEngine:
    """Computes automated statistical summaries and bivariate patterns."""

    @staticmethod
    def get_numerical_summary(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute extended descriptive statistics for all numeric columns.
        Includes Mean, Median, Mode, Std, Variance, Min, 25%, 75%, Max, IQR, Skewness, Kurtosis.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return pd.DataFrame()

        summary_rows = []
        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) == 0:
                continue

            count = len(series)
            mean_val = series.mean()
            median_val = series.median()
            std_val = series.std()
            var_val = series.var()
            min_val = series.min()
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            max_val = series.max()
            p5 = series.quantile(0.05)
            p95 = series.quantile(0.95)

            # Mode
            mode_series = series.mode()
            mode_val = mode_series.iloc[0] if len(mode_series) > 0 else np.nan

            # Skewness & Kurtosis
            skew_val = stats.skew(series, nan_policy="omit") if count > 2 else 0.0
            kurt_val = stats.kurtosis(series, nan_policy="omit") if count > 3 else 0.0

            summary_rows.append({
                "Column": col,
                "Count": count,
                "Mean": round(mean_val, 3),
                "Median": round(median_val, 3),
                "Mode": round(mode_val, 3) if not pd.isna(mode_val) else np.nan,
                "Std Dev": round(std_val, 3),
                "Variance": round(var_val, 3),
                "Min": round(min_val, 3),
                "25% (Q1)": round(q1, 3),
                "75% (Q3)": round(q3, 3),
                "IQR": round(iqr, 3),
                "Max": round(max_val, 3),
                "5th Pct": round(p5, 3),
                "95th Pct": round(p95, 3),
                "Skewness": round(float(skew_val), 3),
                "Kurtosis": round(float(kurt_val), 3)
            })

        return pd.DataFrame(summary_rows)

    @staticmethod
    def get_categorical_summary(df: pd.DataFrame, max_categories: int = 10) -> Dict[str, Any]:
        """
        Compute frequency tables, top/bottom categories, and unique counts for categorical features.
        """
        cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns
        results = {}

        for col in cat_cols:
            series = df[col].astype(str)
            total = len(series)
            val_counts = series.value_counts(dropna=False)
            unique_count = len(val_counts)

            top_cats = val_counts.head(max_categories).to_dict()
            bottom_cats = val_counts.tail(5).to_dict()

            top_pcts = {k: round(v / total * 100, 2) for k, v in top_cats.items()}

            results[col] = {
                "unique_count": unique_count,
                "top_categories": top_cats,
                "top_percentages": top_pcts,
                "bottom_categories": bottom_cats,
                "most_frequent": val_counts.index[0] if len(val_counts) > 0 else "N/A",
                "most_frequent_count": int(val_counts.iloc[0]) if len(val_counts) > 0 else 0,
                "least_frequent": val_counts.index[-1] if len(val_counts) > 0 else "N/A",
                "least_frequent_count": int(val_counts.iloc[-1]) if len(val_counts) > 0 else 0
            }

        return results

    @staticmethod
    def get_correlation_matrix(df: pd.DataFrame, method: str = "pearson") -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Compute correlation matrix and extract significant pairwise correlations.
        Returns: (correlation_df, top_correlated_pairs)
        """
        num_df = df.select_dtypes(include=[np.number])
        if len(num_df.columns) < 2:
            return pd.DataFrame(), []

        corr = num_df.corr(method=method)

        # Extract pairwise combinations without diagonal duplicates
        pairs = []
        cols = corr.columns
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                c1, c2 = cols[i], cols[j]
                val = corr.loc[c1, c2]
                if not pd.isna(val):
                    pairs.append({
                        "feature_1": c1,
                        "feature_2": c2,
                        "correlation": round(float(val), 3),
                        "abs_correlation": round(abs(float(val)), 3),
                        "relationship": "Strong Positive" if val > 0.7 else (
                            "Moderate Positive" if val > 0.4 else (
                                "Strong Negative" if val < -0.7 else (
                                    "Moderate Negative" if val < -0.4 else "Weak / Linear Independence"
                                )
                            )
                        )
                    })

        # Sort by absolute correlation strength descending
        pairs.sort(key=lambda x: x["abs_correlation"], reverse=True)
        return corr, pairs

    @staticmethod
    def get_missing_value_analysis(df: pd.DataFrame) -> pd.DataFrame:
        """Analyze missingness counts and percentages across all columns."""
        total = len(df)
        missing_data = []

        for col in df.columns:
            null_count = int(df[col].isna().sum())
            if null_count > 0:
                missing_data.append({
                    "Column": col,
                    "Data Type": str(df[col].dtype),
                    "Missing Count": null_count,
                    "Missing Percentage": round(null_count / total * 100, 2),
                    "Valid Count": total - null_count
                })

        if not missing_data:
            return pd.DataFrame()

        res_df = pd.DataFrame(missing_data)
        return res_df.sort_values(by="Missing Count", ascending=False).reset_index(drop=True)

