"""
Conversational Query Engine ('Chat With Your Data') for DataSense AI.

Architecture:
  LAYER 1 – Named Business Metric Detection (AOV, Profit Margin, etc.)
  LAYER 2 – Semantic Column Mention Detection (column name in query)
  LAYER 3 – Role-Based Column Resolution (column_roles semantic mapping)
  LAYER 4 – Explicit "Cannot Compute" response (never substitutes a wrong column)

Strictly avoids unsafe code execution (zero unrestricted eval / exec).
"""

import re
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.constants import (
    ROLE_REVENUE, ROLE_PROFIT, ROLE_COST, ROLE_QUANTITY,
    ROLE_PRICE, ROLE_CATEGORY, ROLE_SUBCATEGORY, ROLE_REGION,
    ROLE_PRODUCT, ROLE_DATE, ROLE_CUSTOMER, ROLE_STATUS, ROLE_ID,
    ROLE_DISCOUNT, CHART_COLOR_SEQUENCE
)
from utils.helpers import format_currency, format_number, format_percentage
from backend.visualization_engine import PLOTLY_LAYOUT_DEFAULTS


# ---------------------------------------------------------------------------
# Business Metric Vocabulary
# Phrases that map to composite / derived metric calculations.
# Checked BEFORE generic word matching so multi-word terms win.
# ---------------------------------------------------------------------------
BUSINESS_METRIC_PATTERNS: List[Dict[str, Any]] = [
    # Average Order Value
    {
        "id": "aov",
        "patterns": ["average order value", "aov", "avg order value",
                     "mean order value", "order value"],
        "description": "Average Order Value (AOV) = Total Revenue ÷ Number of Unique Orders",
        "requires": ["revenue", "order_id"],   # logical role names, resolved at runtime
    },
    # Profit Margin
    {
        "id": "profit_margin",
        "patterns": ["profit margin", "gross margin", "margin percentage",
                     "margin %", "margin percent", "net margin"],
        "description": "Profit Margin = (Total Profit ÷ Total Revenue) × 100",
        "requires": ["profit", "revenue"],
    },
    # Revenue per Unit / per Item
    {
        "id": "revenue_per_unit",
        "patterns": ["revenue per unit", "revenue per item", "sales per unit",
                     "average price", "avg price", "mean price"],
        "description": "Revenue per Unit = Total Revenue ÷ Total Quantity",
        "requires": ["revenue", "quantity"],
    },
    # Customer Lifetime Value (simple proxy)
    {
        "id": "clv",
        "patterns": ["customer lifetime value", "clv", "ltv",
                     "lifetime value", "customer value"],
        "description": "CLV (Proxy) = Total Revenue ÷ Number of Unique Customers",
        "requires": ["revenue", "customer"],
    },
    # Attrition / Churn Rate
    {
        "id": "attrition_rate",
        "patterns": ["attrition rate", "churn rate", "attrition", "churn",
                     "turnover rate", "employee turnover"],
        "description": "Attrition Rate = (Attrition Count ÷ Total Employees) × 100",
        "requires": ["attrition"],
    },
]

# Keyword → semantic role mapping used for column resolution
# Ordered from most-specific (multi-word) to least-specific (single-word)
# to prevent "age" matching inside "average"
TERM_TO_ROLES: List[Tuple[str, List[str]]] = [
    # Revenue / Sales
    ("total revenue",   [ROLE_REVENUE]),
    ("total sales",     [ROLE_REVENUE]),
    ("gross revenue",   [ROLE_REVENUE]),
    ("net revenue",     [ROLE_REVENUE]),
    ("revenue",         [ROLE_REVENUE]),
    ("sales amount",    [ROLE_REVENUE]),
    ("sales",           [ROLE_REVENUE]),
    ("turnover",        [ROLE_REVENUE]),
    # Profit
    ("total profit",    [ROLE_PROFIT]),
    ("net profit",      [ROLE_PROFIT]),
    ("gross profit",    [ROLE_PROFIT]),
    ("profit",          [ROLE_PROFIT]),
    ("earnings",        [ROLE_PROFIT]),
    # Cost
    ("total cost",      [ROLE_COST]),
    ("operating cost",  [ROLE_COST]),
    ("cost",            [ROLE_COST]),
    ("expense",         [ROLE_COST]),
    ("expenditure",     [ROLE_COST]),
    # Quantity
    ("units sold",      [ROLE_QUANTITY]),
    ("quantity sold",   [ROLE_QUANTITY]),
    ("quantity",        [ROLE_QUANTITY]),
    ("volume",          [ROLE_QUANTITY]),
    ("units",           [ROLE_QUANTITY]),
    # Price
    ("unit price",      [ROLE_PRICE]),
    ("selling price",   [ROLE_PRICE]),
    ("price",           [ROLE_PRICE]),
    # Discount (has its own role — do NOT map to ROLE_COST or it hijacks cost queries)
    ("discount",        [ROLE_DISCOUNT]),
    # Category / Product / Region / Customer
    ("category",        [ROLE_CATEGORY]),
    ("product",         [ROLE_PRODUCT]),
    ("region",          [ROLE_REGION]),
    ("area",            [ROLE_REGION]),
    ("zone",            [ROLE_REGION]),
    ("territory",       [ROLE_REGION]),
    ("customer",        [ROLE_CUSTOMER]),
    ("client",          [ROLE_CUSTOMER]),
    ("segment",         [ROLE_CATEGORY]),
    # Date / Time
    ("date",            [ROLE_DATE]),
    ("month",           [ROLE_DATE]),
    ("year",            [ROLE_DATE]),
    ("quarter",         [ROLE_DATE]),
    ("time",            [ROLE_DATE]),
    # Order ID (identifier)
    ("order id",        [ROLE_ID]),
    ("order_id",        [ROLE_ID]),
    ("transaction",     [ROLE_ID]),
    ("invoice",         [ROLE_ID]),
]

# Words that should NEVER be used as a fallback numeric metric
# (demographics / identifiers / internal codes)
NON_METRIC_COLUMN_HINTS = [
    "age", "id", "code", "zip", "postal", "phone", "index",
    "row", "sequence", "number", "no", "num", "key", "rank",
    "year_of_birth", "dob", "ssn", "employee_id", "customer_id",
    "order_id", "product_id", "latitude", "longitude", "lat", "lon",
]


class ChatEngine:
    """
    Safe, deterministic natural language query processor for tabular datasets.

    Query interpretation pipeline (in priority order):
      1. Named business metric (AOV, Profit Margin, CLV, …)
      2. Summary / overview
      3. Trend / time-series
      4. Ranking (top / bottom)
      5. Aggregation (average / total / sum) with explicit column resolution
      6. Correlation / relationship
      7. Count / headcount
      8. Generic column fallback
    """

    def __init__(self, df: pd.DataFrame, column_roles: Dict[str, Dict[str, Any]]):
        self.df = df
        self.column_roles = column_roles
        # Build word-boundary safe lower-case column map
        self.cols_lower_map: Dict[str, str] = {str(c).lower(): str(c) for c in df.columns}

    # =========================================================================
    # Public Entry Point
    # =========================================================================
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a natural language question and return a structured response."""
        if self.df.empty:
            return {
                "answer": "The dataset is empty. Please upload or load a dataset first.",
                "data": None, "figure": None, "status": "empty"
            }

        q = query.strip().lower()

        # --- LAYER 1: Named composite business metrics (checked first) ---
        metric_match = self._match_business_metric(q)
        if metric_match:
            return self._handle_named_metric(metric_match, q)

        # --- Standard intent routing ---
        if any(w in q for w in ["summary", "overview", "major finding",
                                 "key findings", "what is this dataset", "summarize"]):
            return self._handle_summary_query()

        if any(w in q for w in ["trend", "over time", "timeline",
                                  "monthly", "historical", "daily", "weekly"]):
            return self._handle_trend_query(q)

        if any(w in q for w in ["highest", "top", "best", "most",
                                  "maximum", "max", "leader", "largest"]):
            return self._handle_ranking_query(q, ascending=False)

        if any(w in q for w in ["lowest", "bottom", "worst", "least",
                                  "minimum", "min", "smallest"]):
            return self._handle_ranking_query(q, ascending=True)

        if any(w in q for w in ["average", "mean", "avg"]):
            return self._handle_aggregation_query(q, agg_type="mean")

        if any(w in q for w in ["total", "sum", "overall", "aggregate"]):
            return self._handle_aggregation_query(q, agg_type="sum")

        if any(w in q for w in ["correlation", "relationship", "correlated", "relate"]):
            return self._handle_correlation_query(q)

        if any(w in q for w in ["how many", "count", "number of", "headcount"]):
            return self._handle_count_query(q)

        return self._handle_generic_column_query(q)

    # =========================================================================
    # LAYER 1 — Named Business Metric Handlers
    # =========================================================================
    def _match_business_metric(self, q: str) -> Optional[Dict[str, Any]]:
        """Return the first matching business metric definition or None."""
        for metric in BUSINESS_METRIC_PATTERNS:
            for pattern in metric["patterns"]:
                if pattern in q:
                    return metric
        return None

    def _handle_named_metric(self, metric: Dict[str, Any], q: str) -> Dict[str, Any]:
        """Dispatch to the correct named metric calculator."""
        mid = metric["id"]
        if mid == "aov":
            return self._calc_aov()
        if mid == "profit_margin":
            return self._calc_profit_margin()
        if mid == "revenue_per_unit":
            return self._calc_revenue_per_unit()
        if mid == "clv":
            return self._calc_clv()
        if mid == "attrition_rate":
            return self._calc_attrition_rate()
        return {"answer": f"Metric '{mid}' is recognized but not yet implemented.",
                "data": None, "figure": None}

    # --- AOV ------------------------------------------------------------------
    def _calc_aov(self) -> Dict[str, Any]:
        rev_col = self._find_column_by_roles([ROLE_REVENUE])
        order_col = self._find_column_by_roles([ROLE_ID])  # Order ID

        if rev_col is None:
            return self._cannot_compute(
                "Average Order Value",
                "Formula: Total Revenue ÷ Number of Unique Orders",
                missing="a Revenue column (e.g. Sales, Revenue, Amount)"
            )

        total_rev = self.df[rev_col].dropna().sum()

        if order_col is not None:
            n_orders = self.df[order_col].dropna().nunique()
            calc_method = f"Total Revenue ({rev_col}) ÷ Unique Orders ({order_col})"
        else:
            # Fallback: treat each row as one order
            n_orders = len(self.df[rev_col].dropna())
            calc_method = f"Total Revenue ({rev_col}) ÷ Total Transactions (each row = 1 order)"

        aov = total_rev / n_orders if n_orders > 0 else 0.0

        summary_df = pd.DataFrame([
            {"Metric": "Total Revenue", "Value": format_currency(total_rev)},
            {"Metric": "Number of Orders", "Value": f"{n_orders:,}"},
            {"Metric": "Average Order Value (AOV)", "Value": format_currency(aov)},
        ])

        answer = (
            f"**Average Order Value (AOV) = {format_currency(aov)}**\n\n"
            f"**Formula:** {calc_method}\n\n"
            f"- Total Revenue (`{rev_col}`): **{format_currency(total_rev)}**\n"
            f"- Number of Orders: **{n_orders:,}**\n"
            f"- AOV: **{format_currency(aov)}**"
        )
        return {"answer": answer, "data": summary_df, "figure": None,
                "explanation": f"AOV calculated using {calc_method}"}

    # --- Profit Margin --------------------------------------------------------
    def _calc_profit_margin(self) -> Dict[str, Any]:
        rev_col = self._find_column_by_roles([ROLE_REVENUE])
        profit_col = self._find_column_by_roles([ROLE_PROFIT])

        missing = []
        if rev_col is None:
            missing.append("a Revenue column")
        if profit_col is None:
            missing.append("a Profit column")
        if missing:
            return self._cannot_compute(
                "Profit Margin",
                "Formula: (Total Profit ÷ Total Revenue) × 100",
                missing=" and ".join(missing)
            )

        total_rev = self.df[rev_col].dropna().sum()
        total_profit = self.df[profit_col].dropna().sum()
        margin = (total_profit / total_rev * 100) if total_rev != 0 else 0.0

        summary_df = pd.DataFrame([
            {"Metric": "Total Revenue", "Value": format_currency(total_rev)},
            {"Metric": "Total Profit", "Value": format_currency(total_profit)},
            {"Metric": "Profit Margin", "Value": f"{margin:.2f}%"},
        ])

        answer = (
            f"**Profit Margin = {margin:.2f}%**\n\n"
            f"**Formula:** (Total Profit ÷ Total Revenue) × 100\n\n"
            f"- Total Profit (`{profit_col}`): **{format_currency(total_profit)}**\n"
            f"- Total Revenue (`{rev_col}`): **{format_currency(total_rev)}**\n"
            f"- Profit Margin: **{margin:.2f}%**"
        )
        return {"answer": answer, "data": summary_df, "figure": None}

    # --- Revenue per Unit -----------------------------------------------------
    def _calc_revenue_per_unit(self) -> Dict[str, Any]:
        rev_col = self._find_column_by_roles([ROLE_REVENUE])
        qty_col = self._find_column_by_roles([ROLE_QUANTITY])

        # Fallback: use price column directly
        if rev_col and qty_col is None:
            price_col = self._find_column_by_roles([ROLE_PRICE])
            if price_col:
                avg_price = self.df[price_col].dropna().mean()
                summary_df = pd.DataFrame([
                    {"Metric": f"Average {price_col}", "Value": format_currency(avg_price)}
                ])
                return {
                    "answer": (
                        f"**Average Unit Price = {format_currency(avg_price)}**\n\n"
                        f"Computed as the mean of the `{price_col}` column "
                        f"({len(self.df[price_col].dropna()):,} records)."
                    ),
                    "data": summary_df, "figure": None
                }

        if rev_col is None or qty_col is None:
            return self._cannot_compute(
                "Revenue per Unit",
                "Formula: Total Revenue ÷ Total Quantity",
                missing="Revenue and/or Quantity columns"
            )

        total_rev = self.df[rev_col].dropna().sum()
        total_qty = self.df[qty_col].dropna().sum()
        rpu = total_rev / total_qty if total_qty != 0 else 0.0

        summary_df = pd.DataFrame([
            {"Metric": "Total Revenue", "Value": format_currency(total_rev)},
            {"Metric": "Total Quantity", "Value": format_number(total_qty)},
            {"Metric": "Revenue per Unit", "Value": format_currency(rpu)},
        ])
        answer = (
            f"**Revenue per Unit = {format_currency(rpu)}**\n\n"
            f"- Total Revenue (`{rev_col}`): **{format_currency(total_rev)}**\n"
            f"- Total Quantity (`{qty_col}`): **{format_number(total_qty)}**"
        )
        return {"answer": answer, "data": summary_df, "figure": None}

    # --- CLV ------------------------------------------------------------------
    def _calc_clv(self) -> Dict[str, Any]:
        rev_col = self._find_column_by_roles([ROLE_REVENUE])
        cust_col = self._find_column_by_roles([ROLE_CUSTOMER])

        if rev_col is None:
            return self._cannot_compute(
                "Customer Lifetime Value",
                "Formula: Total Revenue ÷ Unique Customers",
                missing="a Revenue column"
            )

        total_rev = self.df[rev_col].dropna().sum()
        if cust_col:
            n_customers = self.df[cust_col].dropna().nunique()
        else:
            n_customers = len(self.df)
            cust_col = "row count (no customer ID column found)"

        clv = total_rev / n_customers if n_customers > 0 else 0.0

        summary_df = pd.DataFrame([
            {"Metric": "Total Revenue", "Value": format_currency(total_rev)},
            {"Metric": "Unique Customers", "Value": f"{n_customers:,}"},
            {"Metric": "CLV (Proxy)", "Value": format_currency(clv)},
        ])
        answer = (
            f"**Customer Lifetime Value (Proxy) = {format_currency(clv)}**\n\n"
            f"- Total Revenue (`{rev_col}`): **{format_currency(total_rev)}**\n"
            f"- Unique Customers: **{n_customers:,}** (from `{cust_col}`)"
        )
        return {"answer": answer, "data": summary_df, "figure": None}

    # --- Attrition Rate -------------------------------------------------------
    def _calc_attrition_rate(self) -> Dict[str, Any]:
        # Look for an Attrition / left column
        attrition_col = None
        for col in self.df.columns:
            if any(w in col.lower() for w in ["attrition", "churn", "left", "resigned", "terminated"]):
                attrition_col = col
                break

        if attrition_col is None:
            return self._cannot_compute(
                "Attrition Rate",
                "Formula: (Employees who Left ÷ Total Employees) × 100",
                missing="an Attrition/Churn column (e.g. Attrition, Churn, Left)"
            )

        series = self.df[attrition_col]
        # Handle Yes/No, True/False, 1/0 formats
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            left_count = int(series.str.strip().str.lower().isin(["yes", "true", "1", "y"]).sum())
        elif pd.api.types.is_bool_dtype(series):
            left_count = int(series.sum())
        else:
            left_count = int((series == 1).sum())

        total = len(series.dropna())
        rate = (left_count / total * 100) if total > 0 else 0.0

        summary_df = pd.DataFrame([
            {"Metric": "Total Employees", "Value": f"{total:,}"},
            {"Metric": "Employees Left", "Value": f"{left_count:,}"},
            {"Metric": "Attrition Rate", "Value": f"{rate:.2f}%"},
        ])
        answer = (
            f"**Attrition Rate = {rate:.2f}%**\n\n"
            f"**Formula:** (Employees Left ÷ Total) × 100\n\n"
            f"- Employees left (`{attrition_col}`): **{left_count:,}**\n"
            f"- Total employees: **{total:,}**"
        )
        return {"answer": answer, "data": summary_df, "figure": None}

    # =========================================================================
    # Standard Intent Handlers
    # =========================================================================
    def _handle_summary_query(self) -> Dict[str, Any]:
        """Dataset executive overview."""
        n_rows = len(self.df)
        n_cols = len(self.df.columns)
        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = self.df.select_dtypes(include=["object", "category", "string"]).columns.tolist()

        answer = (
            f"This dataset contains **{n_rows:,} rows** and **{n_cols} columns** "
            f"({len(num_cols)} numerical and {len(cat_cols)} categorical). "
            f"Columns: {', '.join(self.df.columns[:8])}{'…' if n_cols > 8 else ''}."
        )
        return {"answer": answer, "data": self.df.head(5), "figure": None,
                "explanation": "Executive snapshot of dataset structure."}

    def _handle_trend_query(self, q: str) -> Dict[str, Any]:
        """Analyze a metric over time."""
        date_col = self._resolve_column_smart(q, [ROLE_DATE])
        metric_col = self._resolve_column_smart(q, [ROLE_REVENUE, ROLE_PROFIT, ROLE_COST, ROLE_QUANTITY],
                                                 require_numeric=True)

        if not date_col:
            return {"answer": "Could not identify a Date column in the dataset to build a trend chart.",
                    "data": None, "figure": None}

        if not metric_col:
            # Pick the best available revenue/profit column, never random
            metric_col = (self._find_column_by_roles([ROLE_REVENUE])
                          or self._find_column_by_roles([ROLE_PROFIT]))
            if not metric_col:
                return {"answer": f"Found date column `{date_col}` but no revenue or profit metric column to trend.",
                        "data": None, "figure": None}

        temp_df = self.df[[date_col, metric_col]].dropna().copy()
        temp_df[date_col] = pd.to_datetime(temp_df[date_col], errors="coerce")
        temp_df = temp_df.dropna().sort_values(by=date_col)

        if len(temp_df) > 90:
            trend_df = temp_df.set_index(date_col).resample("ME").sum().reset_index()
            x_title = "Month"
        else:
            trend_df = temp_df.groupby(date_col)[metric_col].sum().reset_index()
            x_title = "Date"

        fig = px.area(trend_df, x=date_col, y=metric_col,
                      title=f"Trend: {metric_col} Over Time",
                      color_discrete_sequence=["#4F46E5"])
        fig.update_layout(**PLOTLY_LAYOUT_DEFAULTS)

        total_metric = trend_df[metric_col].sum()
        is_currency = any(k in metric_col.lower() for k in ["sales", "revenue", "profit", "amount", "cost"])
        val_str = format_currency(total_metric) if is_currency else format_number(total_metric)

        answer = (
            f"Trend for **{metric_col}** over time. "
            f"Cumulative total: **{val_str}**."
        )
        return {"answer": answer, "data": trend_df.head(10), "figure": fig}

    def _handle_ranking_query(self, q: str, ascending: bool = False) -> Dict[str, Any]:
        """Find top / bottom performing dimension-metric combinations."""
        dim_col = self._resolve_column_smart(q,
                   [ROLE_CATEGORY, ROLE_REGION, ROLE_PRODUCT, ROLE_CUSTOMER, ROLE_SUBCATEGORY])
        metric_col = self._resolve_column_smart(q,
                     [ROLE_REVENUE, ROLE_PROFIT, ROLE_COST, ROLE_QUANTITY],
                     require_numeric=True)

        # Dimension fallback: pick first categorical column
        if not dim_col:
            cat_candidates = self.df.select_dtypes(include=["object", "category"]).columns
            dim_col = cat_candidates[0] if len(cat_candidates) > 0 else None
        if not dim_col:
            return {"answer": "Could not find a categorical dimension column to rank.",
                    "data": None, "figure": None}

        # Metric fallback: pick revenue/profit role if not found by name
        if not metric_col:
            metric_col = (self._find_column_by_roles([ROLE_REVENUE])
                          or self._find_column_by_roles([ROLE_PROFIT])
                          or self._find_column_by_roles([ROLE_COST]))

        if not metric_col:
            # Rank by frequency
            ranked = self.df[dim_col].value_counts().reset_index()
            ranked.columns = [dim_col, "Count"]
            top = ranked.iloc[0]
            adj = "lowest" if ascending else "highest"
            return {"answer": f"The **{dim_col}** with {adj} frequency: **'{top[dim_col]}'** ({top['Count']:,} records).",
                    "data": ranked.head(10), "figure": None}

        n_match = re.search(r"top\s+(\d+)|bottom\s+(\d+)", q)
        top_n = int((n_match.group(1) or n_match.group(2))) if n_match else 5
        top_n = min(20, max(1, top_n))

        valid_df = self.df[[dim_col, metric_col]].dropna()
        ranked = (valid_df.groupby(dim_col)[metric_col]
                  .sum().reset_index()
                  .sort_values(by=metric_col, ascending=ascending))

        if len(ranked) == 0:
            return {"answer": "No valid data points found.", "data": None, "figure": None}

        leader = ranked.iloc[0]
        rank_type = "lowest" if ascending else "highest"
        is_currency = any(k in metric_col.lower() for k in ["sales", "revenue", "profit", "amount", "cost", "price"])
        val_str = format_currency(leader[metric_col]) if is_currency else format_number(leader[metric_col])

        answer = (
            f"The **{dim_col}** with the {rank_type} **{metric_col}** is "
            f"**'{leader[dim_col]}'** with **{val_str}**."
        )

        plot_df = ranked.head(top_n) if not ascending else ranked.tail(top_n)
        fig = px.bar(plot_df, x=dim_col, y=metric_col,
                     color=metric_col,
                     color_continuous_scale="Reds" if ascending else "Viridis",
                     title=f"{'Top' if not ascending else 'Bottom'} {len(plot_df)} {dim_col} by {metric_col}")
        fig.update_layout(**PLOTLY_LAYOUT_DEFAULTS)

        return {"answer": answer, "data": ranked.head(10), "figure": fig,
                "explanation": f"Summed {metric_col} by {dim_col}."}

    def _handle_aggregation_query(self, q: str, agg_type: str = "mean") -> Dict[str, Any]:
        """
        Compute mean/sum/total for a metric column.

        Resolution order:
          1. Explicit column name in query (word-boundary matched)
          2. Business term → role mapping from TERM_TO_ROLES
          3. Column role mapping from column_roles dict
          4. → Return "cannot compute" (never picks a random column)
        """
        metric_col = self._resolve_column_smart(
            q,
            [ROLE_REVENUE, ROLE_PROFIT, ROLE_COST, ROLE_PRICE, ROLE_QUANTITY],
            require_numeric=True
        )

        if not metric_col:
            # Give a targeted error — explain what metric was asked and what's available
            return self._aggregation_not_found(q, agg_type)

        # Check for grouping intent
        has_grouping = any(w in q for w in [" by ", " per ", " across ", " each ", " for each ", " grouped"])
        dim_col = None

        if has_grouping:
            dim_col = self._resolve_column_smart(
                q,
                [ROLE_CATEGORY, ROLE_REGION, ROLE_PRODUCT, ROLE_SUBCATEGORY],
                exclude_cols=[metric_col]
            )
        else:
            # Check if a categorical column was explicitly mentioned (word-boundary)
            for col in self.df.columns:
                if col == metric_col:
                    continue
                if not pd.api.types.is_numeric_dtype(self.df[col]):
                    if re.search(r'\b' + re.escape(col.lower()) + r'\b', q):
                        dim_col = col
                        break

        if dim_col and not pd.api.types.is_numeric_dtype(self.df[dim_col]):
            grouped = (self.df.groupby(dim_col)[metric_col]
                       .agg(agg_type).reset_index()
                       .sort_values(by=metric_col, ascending=False))
            leader = grouped.iloc[0]
            is_currency = any(k in metric_col.lower() for k in
                              ["sales", "revenue", "profit", "amount", "cost", "price"])
            val_str = format_currency(leader[metric_col]) if is_currency else format_number(leader[metric_col])
            answer = (
                f"{agg_type.capitalize()} **{metric_col}** by **{dim_col}**: "
                f"Top group is **'{leader[dim_col]}'** ({val_str})."
            )
            fig = px.bar(grouped.head(10), x=dim_col, y=metric_col,
                         title=f"{agg_type.capitalize()} {metric_col} by {dim_col}")
            fig.update_layout(**PLOTLY_LAYOUT_DEFAULTS)
            return {"answer": answer, "data": grouped.head(10), "figure": fig}

        # Overall scalar calculation
        series = self.df[metric_col].dropna()
        if agg_type == "mean":
            val, op_label = series.mean(), "average"
        elif agg_type == "sum":
            val, op_label = series.sum(), "total"
        else:
            val, op_label = series.median(), "median"

        is_currency = any(k in metric_col.lower() for k in
                          ["sales", "revenue", "profit", "salary", "cost", "price", "amount"])
        val_str = format_currency(val) if is_currency else format_number(val)

        answer = f"The **{op_label} {metric_col}** = **{val_str}** (across {len(series):,} records)."
        return {
            "answer": answer,
            "data": pd.DataFrame([
                {"Metric": f"{op_label.capitalize()} {metric_col}", "Value": val_str,
                 "Records Used": f"{len(series):,}"}
            ]),
            "figure": None
        }

    def _handle_count_query(self, q: str) -> Dict[str, Any]:
        """Count unique values or total rows."""
        # If explicitly asking about records/rows/entries: count rows
        if any(w in q for w in ["records", "rows", "entries", "transactions", "observations"]):
            total_rows = len(self.df)
            answer = f"The dataset contains **{total_rows:,} total records**."
            return {"answer": answer,
                    "data": pd.DataFrame([{"Total Rows": total_rows}]),
                    "figure": None}

        target_col = self._resolve_column_smart(q, [ROLE_CUSTOMER, ROLE_ID, ROLE_PRODUCT, ROLE_CATEGORY])
        if target_col:
            n_unique = self.df[target_col].nunique(dropna=True)
            answer = f"There are **{n_unique:,} unique `{target_col}`** values in the dataset."
            return {"answer": answer,
                    "data": pd.DataFrame([{"Field": target_col, "Unique Count": n_unique}]),
                    "figure": None}

        total_rows = len(self.df)
        answer = f"The dataset contains **{total_rows:,} total records**."
        return {"answer": answer,
                "data": pd.DataFrame([{"Total Rows": total_rows}]),
                "figure": None}

    def _handle_correlation_query(self, q: str) -> Dict[str, Any]:
        """Compute bivariate Pearson correlation."""
        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) < 2:
            return {"answer": "Need at least two numeric columns for correlation analysis.",
                    "data": None, "figure": None}

        # Word-boundary match to avoid "age" in "average"
        mentioned = [c for c in num_cols
                     if re.search(r'\b' + re.escape(str(c).lower()) + r'\b', q)]
        if len(mentioned) >= 2:
            c1, c2 = mentioned[0], mentioned[1]
        else:
            c1, c2 = num_cols[0], num_cols[1]

        valid_df = self.df[[c1, c2]].dropna()
        corr_val = valid_df[c1].corr(valid_df[c2])

        fig = px.scatter(valid_df, x=c1, y=c2, trendline="ols",
                         title=f"Correlation: {c1} vs {c2}  (r = {corr_val:.2f})")
        fig.update_layout(**PLOTLY_LAYOUT_DEFAULTS)

        strength = ("strong positive" if corr_val > 0.6 else
                    "moderate positive" if corr_val > 0.3 else
                    "strong negative" if corr_val < -0.6 else
                    "moderate negative" if corr_val < -0.3 else
                    "weak / negligible")
        answer = (
            f"Pearson r between **{c1}** and **{c2}**: **{corr_val:.3f}** "
            f"({strength} association)."
        )
        return {"answer": answer,
                "data": pd.DataFrame([{"Feature A": c1, "Feature B": c2, "Pearson r": round(corr_val, 3)}]),
                "figure": fig}

    def _handle_generic_column_query(self, q: str) -> Dict[str, Any]:
        """Fallback: match explicit column name mentions using word boundaries."""
        for col_name in self.df.columns:
            if re.search(r'\b' + re.escape(str(col_name).lower()) + r'\b', q):
                if pd.api.types.is_numeric_dtype(self.df[col_name]):
                    s = self.df[col_name].dropna()
                    answer = (
                        f"**{col_name}** statistics:\n"
                        f"- Mean: **{format_number(s.mean())}**\n"
                        f"- Median: **{format_number(s.median())}**\n"
                        f"- Min / Max: **{format_number(s.min())}** / **{format_number(s.max())}**\n"
                        f"- Std Dev: **{format_number(s.std())}**"
                    )
                    fig = px.histogram(self.df, x=col_name, title=f"Distribution of {col_name}",
                                       color_discrete_sequence=["#4F46E5"])
                    fig.update_layout(**PLOTLY_LAYOUT_DEFAULTS)
                    return {"answer": answer, "data": s.describe().to_frame().reset_index(), "figure": fig}
                else:
                    top_vals = self.df[col_name].value_counts().head(8)
                    answer = f"Top values in **{col_name}**: {', '.join([f'{k} ({v})' for k, v in top_vals.items()])}."
                    return {"answer": answer, "data": top_vals.reset_index(), "figure": None}

        return {
            "answer": (
                "I couldn't match your question to a specific column or business metric.\n\n"
                "**Try questions like:**\n"
                "- *'What is the Average Order Value?'*\n"
                "- *'What is the total revenue?'*\n"
                "- *'Which category has the highest profit?'*\n"
                "- *'Show revenue trends over time'*\n"
                "- *'What is the profit margin?'*\n"
                "- *'What is the correlation between revenue and profit?'*\n\n"
                f"Available columns: `{'`, `'.join(self.df.columns[:12])}{'…' if len(self.df.columns) > 12 else ''}`"
            ),
            "data": None,
            "figure": None
        }

    # =========================================================================
    # Column Resolution — 3-Layer Architecture
    # =========================================================================
    def _resolve_column_smart(
        self,
        query: str,
        preferred_roles: List[str],
        require_numeric: bool = False,
        exclude_cols: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Resolve a column from a natural-language query using three layers:

        1. Exact word-boundary column name match in query text.
        2. Business term → role mapping (TERM_TO_ROLES), phrase-first ordering.
        3. Column role mapping from column_roles dict.

        Returns None (never falls back to arbitrary column) if no match found.
        """
        exclude = {c.lower() for c in (exclude_cols or [])}

        # --- Layer 1: Word-boundary exact column name match ---
        for col in self.df.columns:
            col_lower = str(col).lower()
            if col_lower in exclude:
                continue
            if re.search(r'\b' + re.escape(col_lower) + r'\b', query):
                if require_numeric and not pd.api.types.is_numeric_dtype(self.df[col]):
                    continue
                return col

        # --- Layer 2: Business term → role phrase matching ---
        for term, roles in TERM_TO_ROLES:
            if term not in query:
                continue
            if not any(r in preferred_roles for r in roles):
                continue
            # Find a column mapped to these roles
            for role in roles:
                col = self._find_column_by_roles([role], exclude=exclude,
                                                  require_numeric=require_numeric)
                if col:
                    return col

        # --- Layer 3: Semantic role mapping from column_roles dict ---
        # ONLY apply when user didn't mention a specific known business term
        # (prevents e.g. "discount" or "cost" query falling back to Revenue)
        has_specific_term = any(term in query for term, _ in TERM_TO_ROLES)
        if not has_specific_term:
            for role in preferred_roles:
                col = self._find_column_by_roles([role], exclude=exclude,
                                                  require_numeric=require_numeric)
                if col:
                    return col

        return None  # Explicit None — never fall through to a random column

    def _find_column_by_roles(
        self,
        roles: List[str],
        exclude: Optional[set] = None,
        require_numeric: bool = False
    ) -> Optional[str]:
        """Find the first column mapped to any of the given semantic roles."""
        exclude = exclude or set()
        for role in roles:
            matched = [
                c for c, meta in self.column_roles.items()
                if meta.get("role") == role
                and c in self.df.columns
                and str(c).lower() not in exclude
            ]
            if require_numeric:
                matched = [c for c in matched if pd.api.types.is_numeric_dtype(self.df[c])]
            if matched:
                return matched[0]
        return None

    # =========================================================================
    # Error Helpers
    # =========================================================================
    def _cannot_compute(self, metric_name: str, formula: str, missing: str) -> Dict[str, Any]:
        """Structured 'cannot compute' response — never substitutes a wrong column."""
        available_num = [c for c in self.df.columns if pd.api.types.is_numeric_dtype(self.df[c])]
        return {
            "answer": (
                f"**Cannot compute {metric_name}** for this dataset.\n\n"
                f"**Formula:** {formula}\n\n"
                f"**Missing:** {missing}\n\n"
                f"**Available numeric columns:** `{'`, `'.join(available_num[:10])}`\n\n"
                f"Try uploading a dataset that contains the required columns, "
                f"or check the **Semantic Role Mapping** in Settings to assign roles correctly."
            ),
            "data": None,
            "figure": None
        }

    def _aggregation_not_found(self, q: str, agg_type: str) -> Dict[str, Any]:
        """Explain clearly when no metric column could be matched."""
        # Identify what metric the user seems to be asking about
        asked_about = None
        for term, roles in TERM_TO_ROLES:
            if term in q:
                asked_about = term
                break

        available_num = [c for c in self.df.columns if pd.api.types.is_numeric_dtype(self.df[c])]
        metric_phrase = f"**{asked_about}**" if asked_about else "the requested metric"

        return {
            "answer": (
                f"Could not find a column matching {metric_phrase} in this dataset.\n\n"
                f"**Available numeric columns:** `{'`, `'.join(available_num[:12])}`\n\n"
                f"Try asking with an exact column name, for example:\n"
                + "\n".join([f"- *'What is the {agg_type} {c}?'*" for c in available_num[:4]])
            ),
            "data": None,
            "figure": None
        }
