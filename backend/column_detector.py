"""
Column Detector and Semantic Role Mapping Engine for DataSense AI.
Intelligently infers the business and statistical role of each column in any dataset.
"""

import re
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from utils.constants import (
    ROLE_DATE, ROLE_ID, ROLE_REVENUE, ROLE_COST, ROLE_PROFIT,
    ROLE_QUANTITY, ROLE_PRICE, ROLE_DISCOUNT, ROLE_CATEGORY,
    ROLE_SUBCATEGORY, ROLE_CUSTOMER, ROLE_REGION, ROLE_PRODUCT,
    ROLE_STATUS, ROLE_PERCENTAGE, ROLE_NUMERICAL, ROLE_CATEGORICAL,
    SEMANTIC_ROLES
)


# Keywords and regex patterns for semantic detection
ROLE_PATTERNS = {
    ROLE_DATE: [
        r"(date|dt|time|timestamp|day|month|year|period|hire_date|order_date|ship_date|created_at|updated_at)"
    ],
    ROLE_ID: [
        r"(^id$|_id$|^id_|^pk_|_code$|^code$|uuid|guid|transaction_id|employee_id|order_id|customer_id|sku)"
    ],
    ROLE_REVENUE: [
        r"(revenue|sales|total_sales|total_amount|gross_sales|net_sales|turnover|order_value|transaction_amount|amount|bill_amount)"
    ],
    ROLE_PROFIT: [
        r"(profit|net_profit|gross_profit|margin|earnings|net_income|operating_income)"
    ],
    ROLE_COST: [
        r"(cost|total_cost|expense|cogs|unit_cost|purchase_price|shipping_cost|overhead)"
    ],
    ROLE_PRICE: [
        r"(price|unit_price|selling_price|retail_price|rate|fare|salary|monthly_salary|hourly_rate)"
    ],
    ROLE_QUANTITY: [
        r"(quantity|qty|units|volume|count|items_sold|number_of|absences|annual_absences|headcount)"
    ],
    ROLE_DISCOUNT: [
        r"(discount|rebate|promo_amount|markdown|discount_rate)"
    ],
    ROLE_CUSTOMER: [
        r"(customer|client|buyer|user|account|member|subscriber|patient|shopper)"
    ],
    ROLE_PRODUCT: [
        r"(product|item|sku|service|good|merchandise|offering|product_name)"
    ],
    ROLE_CATEGORY: [
        r"(category|genre|type|department|segment|class|group|sector|industry|division)"
    ],
    ROLE_SUBCATEGORY: [
        r"(subcategory|sub_category|sub_group|sub_dept|sub_type)"
    ],
    ROLE_REGION: [
        r"(region|country|territory|state|city|province|zone|district|location|area|market)"
    ],
    ROLE_STATUS: [
        r"(status|state|condition|stage|phase|attrition|churn|active|is_active|priority|order_priority)"
    ],
    ROLE_PERCENTAGE: [
        r"(percent|percentage|pct|rate|ratio|share|score|satisfaction|rating)"
    ]
}


class ColumnDetector:
    """Infers statistical types and business semantic roles from dataframe columns."""

    def __init__(self, overrides: Optional[Dict[str, str]] = None):
        self.overrides = overrides or {}

    def detect_column_role(self, series: pd.Series, col_name: str) -> Dict[str, Any]:
        """Detect the role, statistical type, and confidence for a single column."""
        # 1. Check if user provided manual override
        if col_name in self.overrides and self.overrides[col_name] in SEMANTIC_ROLES:
            return {
                "column": col_name,
                "role": self.overrides[col_name],
                "confidence": 1.0,
                "statistical_type": self._get_stat_type(series),
                "is_override": True
            }

        norm_name = str(col_name).strip().lower().replace(" ", "_").replace("-", "_")
        stat_type = self._get_stat_type(series)
        n_unique = series.nunique(dropna=True)
        total_valid = series.count()
        uniqueness_ratio = (n_unique / total_valid) if total_valid > 0 else 0

        # Check for Date first
        if stat_type == "datetime" or self._matches_pattern(norm_name, ROLE_PATTERNS[ROLE_DATE]):
            if self._can_be_datetime(series):
                return {
                    "column": col_name,
                    "role": ROLE_DATE,
                    "confidence": 0.95 if stat_type == "datetime" else 0.85,
                    "statistical_type": "datetime",
                    "is_override": False
                }

        # Check for ID (high uniqueness and ID-like name)
        if self._matches_pattern(norm_name, ROLE_PATTERNS[ROLE_ID]) or (uniqueness_ratio > 0.90 and n_unique > 50):
            if not pd.api.types.is_float_dtype(series):
                return {
                    "column": col_name,
                    "role": ROLE_ID,
                    "confidence": 0.90 if self._matches_pattern(norm_name, ROLE_PATTERNS[ROLE_ID]) else 0.70,
                    "statistical_type": "identifier",
                    "is_override": False
                }

        # Check for Revenue / Profit / Cost / Price / Discount / Quantity
        if stat_type == "numeric":
            # Check Profit first (often has negative values)
            if self._matches_pattern(norm_name, ROLE_PATTERNS[ROLE_PROFIT]):
                return {
                    "column": col_name,
                    "role": ROLE_PROFIT,
                    "confidence": 0.90,
                    "statistical_type": "numeric",
                    "is_override": False
                }
            # Check Revenue
            if self._matches_pattern(norm_name, ROLE_PATTERNS[ROLE_REVENUE]):
                return {
                    "column": col_name,
                    "role": ROLE_REVENUE,
                    "confidence": 0.90,
                    "statistical_type": "numeric",
                    "is_override": False
                }
            # Check Cost
            if self._matches_pattern(norm_name, ROLE_PATTERNS[ROLE_COST]):
                return {
                    "column": col_name,
                    "role": ROLE_COST,
                    "confidence": 0.90,
                    "statistical_type": "numeric",
                    "is_override": False
                }
            # Check Price
            if self._matches_pattern(norm_name, ROLE_PATTERNS[ROLE_PRICE]):
                return {
                    "column": col_name,
                    "role": ROLE_PRICE,
                    "confidence": 0.85,
                    "statistical_type": "numeric",
                    "is_override": False
                }
            # Check Quantity
            if self._matches_pattern(norm_name, ROLE_PATTERNS[ROLE_QUANTITY]) or (pd.api.types.is_integer_dtype(series) and series.min() >= 0 and series.max() < 100000):
                if self._matches_pattern(norm_name, ROLE_PATTERNS[ROLE_QUANTITY]):
                    return {
                        "column": col_name,
                        "role": ROLE_QUANTITY,
                        "confidence": 0.90,
                        "statistical_type": "numeric",
                        "is_override": False
                    }
            # Check Discount / Percentage
            if self._matches_pattern(norm_name, ROLE_PATTERNS[ROLE_DISCOUNT]) or self._matches_pattern(norm_name, ROLE_PATTERNS[ROLE_PERCENTAGE]):
                return {
                    "column": col_name,
                    "role": ROLE_DISCOUNT if self._matches_pattern(norm_name, ROLE_PATTERNS[ROLE_DISCOUNT]) else ROLE_PERCENTAGE,
                    "confidence": 0.85,
                    "statistical_type": "numeric",
                    "is_override": False
                }
            
            # Default numerical
            return {
                "column": col_name,
                "role": ROLE_NUMERICAL,
                "confidence": 0.70,
                "statistical_type": "numeric",
                "is_override": False
            }

        # Check Categorical roles
        for role, patterns in [
            (ROLE_REGION, ROLE_PATTERNS[ROLE_REGION]),
            (ROLE_CUSTOMER, ROLE_PATTERNS[ROLE_CUSTOMER]),
            (ROLE_PRODUCT, ROLE_PATTERNS[ROLE_PRODUCT]),
            (ROLE_SUBCATEGORY, ROLE_PATTERNS[ROLE_SUBCATEGORY]),
            (ROLE_CATEGORY, ROLE_PATTERNS[ROLE_CATEGORY]),
            (ROLE_STATUS, ROLE_PATTERNS[ROLE_STATUS])
        ]:
            if self._matches_pattern(norm_name, patterns):
                return {
                    "column": col_name,
                    "role": role,
                    "confidence": 0.85,
                    "statistical_type": "categorical",
                    "is_override": False
                }

        # If low cardinality string/object, mark as Category
        if stat_type == "categorical" or (stat_type == "string" and n_unique <= 50):
            return {
                "column": col_name,
                "role": ROLE_CATEGORY,
                "confidence": 0.65,
                "statistical_type": "categorical",
                "is_override": False
            }

        # Fallback
        return {
            "column": col_name,
            "role": ROLE_CATEGORICAL if stat_type in ["string", "object"] else ROLE_NUMERICAL,
            "confidence": 0.50,
            "statistical_type": stat_type,
            "is_override": False
        }

    def detect_all(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Detect roles for all columns in dataframe."""
        results = {}
        for col in df.columns:
            results[col] = self.detect_column_role(df[col], str(col))
        return results

    def get_columns_by_role(self, detected_map: Dict[str, Dict[str, Any]], role: str) -> List[str]:
        """Return all column names matching a given semantic role."""
        return [col for col, meta in detected_map.items() if meta["role"] == role]

    def get_first_column_of_roles(self, detected_map: Dict[str, Dict[str, Any]], roles: List[str]) -> Optional[str]:
        """Find the first matching column across a prioritized list of roles."""
        for role in roles:
            matching = self.get_columns_by_role(detected_map, role)
            if matching:
                return matching[0]
        return None

    def _get_stat_type(self, series: pd.Series) -> str:
        """Determine broad statistical type."""
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
        if pd.api.types.is_numeric_dtype(series):
            return "numeric"
        if pd.api.types.is_bool_dtype(series):
            return "boolean"
        # Check string or object
        return "categorical"

    def _matches_pattern(self, text: str, patterns: List[str]) -> bool:
        """Check if text matches any regex in pattern list."""
        for p in patterns:
            if re.search(p, text, re.IGNORECASE):
                return True
        return False

    def _can_be_datetime(self, series: pd.Series) -> bool:
        """Heuristic to check if series can be parsed as datetime."""
        if pd.api.types.is_datetime64_any_dtype(series):
            return True
        non_null = series.dropna()
        if len(non_null) == 0:
            return False
        # Sample up to 10 non-null values
        sample = non_null.head(10).astype(str)
        try:
            pd.to_datetime(sample, errors="raise")
            return True
        except Exception:
            return False
