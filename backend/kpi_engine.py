"""
Automated Business KPI Identification & Calculation Engine for DataSense AI.
Detects business domain and computes domain-specific executive performance indicators.
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from utils.constants import (
    ROLE_REVENUE, ROLE_PROFIT, ROLE_COST, ROLE_QUANTITY,
    ROLE_PRICE, ROLE_DISCOUNT, ROLE_CUSTOMER, ROLE_ID,
    ROLE_STATUS, ROLE_CATEGORY
)
from utils.helpers import format_currency, format_number, format_percentage


class KPIEngine:
    """Calculates automated domain-aware business KPIs and metrics."""

    @staticmethod
    def identify_domain(df: pd.DataFrame, column_roles: Dict[str, Dict[str, Any]]) -> str:
        """
        Infer business dataset domain:
        - retail_sales
        - ecommerce
        - hr_analytics
        - finance
        - general
        """
        cols_lower = [str(c).lower() for c in df.columns]

        # HR domain
        hr_keywords = ["employee", "department", "salary", "attrition", "hire_date", "performance_rating", "absences", "job_satisfaction"]
        if sum(1 for k in hr_keywords if any(k in c for c in cols_lower)) >= 2:
            return "hr_analytics"

        # E-commerce domain
        ecom_keywords = ["order_id", "ship_date", "customer_segment", "sub_category", "discount_rate", "shipping_cost"]
        if sum(1 for k in ecom_keywords if any(k in c for c in cols_lower)) >= 2:
            return "ecommerce"

        # Retail / Sales domain
        sales_keywords = ["revenue", "sales", "transaction_id", "quantity", "unit_price", "profit", "payment_method", "store"]
        if sum(1 for k in sales_keywords if any(k in c for c in cols_lower)) >= 2:
            return "retail_sales"

        # Finance domain
        finance_keywords = ["cost", "cogs", "operating_expense", "budget", "ebitda", "cash_flow", "asset", "liability"]
        if sum(1 for k in finance_keywords if any(k in c for c in cols_lower)) >= 2:
            return "finance"

        return "general"

    @staticmethod
    def compute_kpis(
        df: pd.DataFrame,
        column_roles: Dict[str, Dict[str, Any]],
        domain: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Compute top business KPIs based on detected domain and column roles.
        """
        if df.empty or len(df.columns) == 0:
            return []

        if not domain:
            domain = KPIEngine.identify_domain(df, column_roles)

        kpis = []

        # Find columns by semantic roles
        rev_col = KPIEngine._find_col_by_role(column_roles, ROLE_REVENUE, df)
        profit_col = KPIEngine._find_col_by_role(column_roles, ROLE_PROFIT, df)
        cost_col = KPIEngine._find_col_by_role(column_roles, ROLE_COST, df)
        qty_col = KPIEngine._find_col_by_role(column_roles, ROLE_QUANTITY, df)
        cust_col = KPIEngine._find_col_by_role(column_roles, ROLE_CUSTOMER, df)
        id_col = KPIEngine._find_col_by_role(column_roles, ROLE_ID, df)
        status_col = KPIEngine._find_col_by_role(column_roles, ROLE_STATUS, df)

        total_records = len(df)

        # -------------------------------------------------------------
        # DOMAIN: HR ANALYTICS
        # -------------------------------------------------------------
        if domain == "hr_analytics":
            # 1. Total Headcount
            kpis.append({
                "id": "hr_headcount",
                "title": "Total Headcount",
                "value": format_number(total_records, decimals=0),
                "raw_value": total_records,
                "icon": "users",
                "subtext": "Active personnel records analyzed",
                "category": "HR"
            })

            # 2. Average Salary
            salary_cols = [c for c in df.columns if any(k in c.lower() for k in ["salary", "compensation", "wage"])]
            if salary_cols and pd.api.types.is_numeric_dtype(df[salary_cols[0]]):
                sal_col = salary_cols[0]
                avg_sal = float(df[sal_col].mean())
                med_sal = float(df[sal_col].median())
                kpis.append({
                    "id": "hr_avg_salary",
                    "title": "Average Salary",
                    "value": format_currency(avg_sal),
                    "raw_value": avg_sal,
                    "icon": "briefcase",
                    "subtext": f"Median: {format_currency(med_sal)}",
                    "category": "HR"
                })

            # 3. Attrition Rate
            attr_cols = [c for c in df.columns if "attrition" in c.lower() or "churn" in c.lower() or "left" in c.lower()]
            if attr_cols:
                attr_series = df[attr_cols[0]].astype(str).str.strip().str.lower()
                yes_count = int(attr_series.isin(["yes", "true", "1", "left"]).sum())
                attr_rate = (yes_count / total_records * 100) if total_records > 0 else 0
                kpis.append({
                    "id": "hr_attrition_rate",
                    "title": "Attrition Rate",
                    "value": format_percentage(attr_rate),
                    "raw_value": attr_rate,
                    "icon": "user-minus",
                    "subtext": f"{yes_count} departures recorded",
                    "delta": f"{attr_rate:.1f}%",
                    "delta_positive": attr_rate < 15.0,
                    "category": "HR"
                })

            # 4. Satisfaction Score
            sat_cols = [c for c in df.columns if "satisfaction" in c.lower() or "engagement" in c.lower()]
            if sat_cols and pd.api.types.is_numeric_dtype(df[sat_cols[0]]):
                avg_sat = float(df[sat_cols[0]].mean())
                kpis.append({
                    "id": "hr_avg_satisfaction",
                    "title": "Avg Job Satisfaction",
                    "value": f"{avg_sat:.1f}/10",
                    "raw_value": avg_sat,
                    "icon": "star",
                    "subtext": "Employee sentiment index",
                    "category": "HR"
                })

            return kpis

        # -------------------------------------------------------------
        # DOMAIN: RETAIL / ECOMMERCE / SALES / FINANCE / GENERAL
        # -------------------------------------------------------------
        # 1. Total Revenue / Gross Sales
        if rev_col and pd.api.types.is_numeric_dtype(df[rev_col]):
            total_rev = float(df[rev_col].sum())
            avg_order = float(df[rev_col].mean())
            kpis.append({
                "id": "total_revenue",
                "title": "Total Revenue",
                "value": format_currency(total_rev),
                "raw_value": total_rev,
                "icon": "dollar-sign",
                "subtext": f"Across {total_records:,} transactions",
                "category": "Financial"
            })

            # 2. Average Order Value (AOV)
            kpis.append({
                "id": "avg_order_value",
                "title": "Average Order Value",
                "value": format_currency(avg_order),
                "raw_value": avg_order,
                "icon": "shopping-cart",
                "subtext": "Mean revenue per transaction",
                "category": "Operational"
            })

        # 3. Total Profit & Margin %
        if profit_col and pd.api.types.is_numeric_dtype(df[profit_col]):
            total_profit = float(df[profit_col].sum())
            kpis.append({
                "id": "total_profit",
                "title": "Total Net Profit",
                "value": format_currency(total_profit),
                "raw_value": total_profit,
                "icon": "trending-up",
                "subtext": "Net bottom-line earnings",
                "delta": "Profitable" if total_profit > 0 else "Operating Loss",
                "delta_positive": total_profit > 0,
                "category": "Financial"
            })

            if rev_col and pd.api.types.is_numeric_dtype(df[rev_col]):
                total_rev = float(df[rev_col].sum())
                margin_pct = (total_profit / total_rev * 100) if total_rev > 0 else 0.0
                kpis.append({
                    "id": "profit_margin",
                    "title": "Gross Margin %",
                    "value": format_percentage(margin_pct),
                    "raw_value": margin_pct,
                    "icon": "percent",
                    "subtext": "Overall operating margin ratio",
                    "delta": f"{margin_pct:.1f}%",
                    "delta_positive": margin_pct >= 15.0,
                    "category": "Financial"
                })

        # 4. Total Units Sold / Volume
        if qty_col and pd.api.types.is_numeric_dtype(df[qty_col]):
            total_units = int(df[qty_col].sum())
            kpis.append({
                "id": "total_volume",
                "title": "Total Volume Sold",
                "value": format_number(total_units, decimals=0),
                "raw_value": total_units,
                "icon": "package",
                "subtext": f"Avg {df[qty_col].mean():.1f} items / transaction",
                "category": "Operational"
            })

        # 5. Customer Reach
        if cust_col:
            unique_customers = int(df[cust_col].nunique())
            kpis.append({
                "id": "unique_customers",
                "title": "Active Customers",
                "value": format_number(unique_customers, decimals=0),
                "raw_value": unique_customers,
                "icon": "users",
                "subtext": f"Repeat ratio: {total_records / unique_customers:.2f}x" if unique_customers > 0 else "",
                "category": "Customer"
            })

        # Fallback if no sales/revenue columns found
        if len(kpis) == 0:
            kpis.append({
                "id": "total_records",
                "title": "Total Observations",
                "value": format_number(total_records, decimals=0),
                "raw_value": total_records,
                "icon": "database",
                "subtext": f"{len(df.columns)} Features Analyzed",
                "category": "Dataset"
            })

            # Check primary numeric column
            num_cols = df.select_dtypes(include=[np.number]).columns
            if len(num_cols) > 0:
                p_col = num_cols[0]
                kpis.append({
                    "id": "primary_sum",
                    "title": f"Total {p_col}",
                    "value": format_number(df[p_col].sum()),
                    "raw_value": float(df[p_col].sum()),
                    "icon": "bar-chart-2",
                    "subtext": f"Avg: {format_number(df[p_col].mean())}",
                    "category": "Statistical"
                })

        return kpis

    @staticmethod
    def _find_col_by_role(column_roles: Dict[str, Dict[str, Any]], target_role: str, df: pd.DataFrame) -> Optional[str]:
        """Lookup first column with given role that exists in df."""
        for col, meta in column_roles.items():
            if meta.get("role") == target_role and col in df.columns:
                return col
        return None

