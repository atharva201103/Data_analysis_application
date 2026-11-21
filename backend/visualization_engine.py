"""
Automated & Custom Data Visualization Engine for DataSense AI.
Generates intelligent Plotly visualizations based on detected column roles,
and supports interactive custom chart construction.
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.constants import (
    CHART_COLOR_SEQUENCE, ROLE_DATE, ROLE_REVENUE, ROLE_PROFIT,
    ROLE_COST, ROLE_QUANTITY, ROLE_CATEGORY, ROLE_REGION,
    ROLE_PRODUCT, ROLE_CUSTOMER, ROLE_STATUS
)


# Standard dark-glass layout styling for Plotly
PLOTLY_LAYOUT_DEFAULTS = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(15, 23, 42, 0.4)",
    plot_bgcolor="rgba(15, 23, 42, 0.2)",
    font=dict(family="Inter, sans-serif", color="#E2E8F0", size=12),
    margin=dict(l=40, r=40, t=50, b=40),
    hoverlabel=dict(bgcolor="#1E293B", font_size=12, font_family="Inter"),
    colorway=CHART_COLOR_SEQUENCE
)


class VisualizationEngine:
    """Generates automated domain-aware charts and custom user-configured plots."""

    @staticmethod
    def generate_recommended_charts(
        df: pd.DataFrame,
        column_roles: Dict[str, Dict[str, Any]],
        max_charts: int = 6
    ) -> List[Dict[str, Any]]:
        """
        Generate intelligent, high-value visualizations based on detected semantic roles.
        Returns a list of dicts: {"title": str, "description": str, "figure": go.Figure, "type": str}
        """
        charts = []
        if df.empty or len(df.columns) == 0:
            return charts

        # Extract role-mapped columns
        date_cols = [c for c, m in column_roles.items() if m.get("role") == ROLE_DATE and c in df.columns]
        rev_cols = [c for c, m in column_roles.items() if m.get("role") == ROLE_REVENUE and c in df.columns]
        profit_cols = [c for c, m in column_roles.items() if m.get("role") == ROLE_PROFIT and c in df.columns]
        cat_cols = [c for c, m in column_roles.items() if m.get("role") in [ROLE_CATEGORY, ROLE_REGION, ROLE_PRODUCT] and c in df.columns]
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # 1. Time Series Trend: Date + (Revenue or Primary Numeric)
        if date_cols and (rev_cols or num_cols):
            date_col = date_cols[0]
            val_col = rev_cols[0] if rev_cols else [c for c in num_cols if c != date_col][0]
            
            try:
                # Group by date
                temp_df = df[[date_col, val_col]].dropna().copy()
                temp_df[date_col] = pd.to_datetime(temp_df[date_col], errors="coerce")
                temp_df = temp_df.dropna().sort_values(by=date_col)
                
                # If many data points, group by month or week
                if len(temp_df) > 90:
                    time_grouped = temp_df.set_index(date_col).resample("ME").sum().reset_index()
                    x_label = "Month"
                else:
                    time_grouped = temp_df.groupby(date_col)[val_col].sum().reset_index()
                    x_label = "Date"

                fig = px.area(
                    time_grouped,
                    x=date_col,
                    y=val_col,
                    title=f"📈 Trend Analysis: {val_col} over Time",
                    color_discrete_sequence=["#4F46E5"]
                )
                fig.update_layout(**PLOTLY_LAYOUT_DEFAULTS)
                fig.update_xaxes(title_text=x_label, gridcolor="rgba(255,255,255,0.08)")
                fig.update_yaxes(title_text=val_col, gridcolor="rgba(255,255,255,0.08)")

                charts.append({
                    "title": f"Timeline Trend: {val_col}",
                    "description": f"Historical trajectory of {val_col} grouped chronologically.",
                    "figure": fig,
                    "type": "time_series"
                })
            except Exception:
                pass

        # 2. Categorical Performance: Category/Region + Revenue or Primary Metric
        if cat_cols and (rev_cols or num_cols):
            cat_col = cat_cols[0]
            val_col = rev_cols[0] if rev_cols else num_cols[0]

            try:
                cat_summary = (
                    df.groupby(cat_col)[val_col]
                    .sum()
                    .reset_index()
                    .sort_values(by=val_col, ascending=False)
                    .head(10)
                )

                fig = px.bar(
                    cat_summary,
                    x=cat_col,
                    y=val_col,
                    color=val_col,
                    color_continuous_scale="Viridis",
                    title=f"📊 {val_col} by {cat_col} (Top 10)"
                )
                fig.update_layout(**PLOTLY_LAYOUT_DEFAULTS)
                fig.update_xaxes(title_text=cat_col, gridcolor="rgba(255,255,255,0.08)")
                fig.update_yaxes(title_text=val_col, gridcolor="rgba(255,255,255,0.08)")

                charts.append({
                    "title": f"{val_col} by {cat_col}",
                    "description": f"Ranking of top-performing {cat_col} segments by {val_col}.",
                    "figure": fig,
                    "type": "categorical_bar"
                })
            except Exception:
                pass

        # 3. Profit vs Revenue Scatter / Correlation
        if rev_cols and profit_cols:
            r_col = rev_cols[0]
            p_col = profit_cols[0]
            hover_name = cat_cols[0] if cat_cols else None

            try:
                fig = px.scatter(
                    df.dropna(subset=[r_col, p_col]),
                    x=r_col,
                    y=p_col,
                    color=hover_name if hover_name and df[hover_name].nunique() <= 10 else None,
                    hover_data=[hover_name] if hover_name else None,
                    title=f"🎯 Profitability Matrix: {p_col} vs {r_col}",
                    trendline="ols" if len(df) > 10 else None
                )
                fig.update_layout(**PLOTLY_LAYOUT_DEFAULTS)
                fig.update_xaxes(title_text=r_col, gridcolor="rgba(255,255,255,0.08)")
                fig.update_yaxes(title_text=p_col, gridcolor="rgba(255,255,255,0.08)")

                charts.append({
                    "title": f"Profitability Matrix: {p_col} vs {r_col}",
                    "description": f"Scatter correlation highlighting margin health and potential margin leakage.",
                    "figure": fig,
                    "type": "scatter_matrix"
                })
            except Exception:
                pass

        # 4. Share of Market / Composition (Donut Chart)
        if len(cat_cols) > 0 and (rev_cols or num_cols):
            cat_col = cat_cols[1] if len(cat_cols) > 1 else cat_cols[0]
            val_col = rev_cols[0] if rev_cols else num_cols[0]

            if df[cat_col].nunique() <= 8:
                try:
                    pie_data = df.groupby(cat_col)[val_col].sum().reset_index()
                    fig = px.pie(
                        pie_data,
                        names=cat_col,
                        values=val_col,
                        hole=0.45,
                        title=f"🍩 {cat_col} Contribution Share ({val_col})",
                        color_discrete_sequence=CHART_COLOR_SEQUENCE
                    )
                    fig.update_layout(**PLOTLY_LAYOUT_DEFAULTS)

                    charts.append({
                        "title": f"{cat_col} Share Breakdown",
                        "description": f"Proportional contribution of each {cat_col} to total {val_col}.",
                        "figure": fig,
                        "type": "donut_composition"
                    })
                except Exception:
                    pass

        # 5. Correlation Heatmap (for numeric features)
        if len(num_cols) >= 3:
            try:
                corr = df[num_cols].corr()
                fig = px.imshow(
                    corr,
                    text_auto=".2f",
                    color_continuous_scale="RdBu_r",
                    zmin=-1,
                    zmax=1,
                    title="🔥 Multi-Feature Correlation Matrix"
                )
                fig.update_layout(**PLOTLY_LAYOUT_DEFAULTS)

                charts.append({
                    "title": "Correlation Matrix",
                    "description": "Pearson correlation coefficient across all numerical variables.",
                    "figure": fig,
                    "type": "heatmap"
                })
            except Exception:
                pass

        # 6. Distribution Analysis (Histogram + Box Plot)
        if num_cols:
            primary_num = rev_cols[0] if rev_cols else num_cols[0]
            try:
                fig = px.histogram(
                    df.dropna(subset=[primary_num]),
                    x=primary_num,
                    marginal="box",
                    nbins=30,
                    title=f"📊 Distribution & Skew: {primary_num}",
                    color_discrete_sequence=["#06B6D4"]
                )
                fig.update_layout(**PLOTLY_LAYOUT_DEFAULTS)
                fig.update_xaxes(title_text=primary_num, gridcolor="rgba(255,255,255,0.08)")
                fig.update_yaxes(title_text="Frequency", gridcolor="rgba(255,255,255,0.08)")

                charts.append({
                    "title": f"Distribution: {primary_num}",
                    "description": f"Statistical distribution and outlier spread for {primary_num}.",
                    "figure": fig,
                    "type": "distribution"
                })
            except Exception:
                pass

        return charts[:max_charts]

    @staticmethod
    def create_custom_chart(
        df: pd.DataFrame,
        chart_type: str,
        x_col: str,
        y_col: Optional[str] = None,
        color_col: Optional[str] = None,
        agg_func: str = "sum",
        barmode: str = "group"
    ) -> go.Figure:
        """
        Build user-configured interactive chart with dynamic aggregation.
        """
        temp_df = df.copy()

        # Handle Aggregations if y_col is numeric and grouped
        if y_col and pd.api.types.is_numeric_dtype(temp_df[y_col]) and chart_type in ["Bar", "Horizontal Bar", "Line", "Area", "Pie / Donut"]:
            group_cols = [x_col]
            if color_col and color_col != "None" and color_col in temp_df.columns and color_col != x_col:
                group_cols.append(color_col)

            if agg_func == "sum":
                plot_data = temp_df.groupby(group_cols)[y_col].sum().reset_index()
            elif agg_func == "mean":
                plot_data = temp_df.groupby(group_cols)[y_col].mean().reset_index()
            elif agg_func == "median":
                plot_data = temp_df.groupby(group_cols)[y_col].median().reset_index()
            elif agg_func == "count":
                plot_data = temp_df.groupby(group_cols)[y_col].count().reset_index()
            elif agg_func == "min":
                plot_data = temp_df.groupby(group_cols)[y_col].min().reset_index()
            elif agg_func == "max":
                plot_data = temp_df.groupby(group_cols)[y_col].max().reset_index()
            else:
                plot_data = temp_df
        else:
            plot_data = temp_df

        color_arg = color_col if color_col and color_col != "None" and color_col in plot_data.columns else None

        if chart_type == "Bar":
            fig = px.bar(
                plot_data,
                x=x_col,
                y=y_col,
                color=color_arg,
                barmode=barmode,
                title=f"{agg_func.capitalize()} of {y_col} by {x_col}" if y_col else f"Count by {x_col}"
            )
        elif chart_type == "Horizontal Bar":
            fig = px.bar(
                plot_data,
                y=x_col,
                x=y_col,
                color=color_arg,
                orientation="h",
                title=f"{agg_func.capitalize()} of {y_col} by {x_col}"
            )
        elif chart_type == "Line":
            fig = px.line(
                plot_data.sort_values(by=x_col),
                x=x_col,
                y=y_col,
                color=color_arg,
                markers=True,
                title=f"{y_col} Trend across {x_col}"
            )
        elif chart_type == "Area":
            fig = px.area(
                plot_data.sort_values(by=x_col),
                x=x_col,
                y=y_col,
                color=color_arg,
                title=f"Cumulative {y_col} across {x_col}"
            )
        elif chart_type == "Scatter":
            fig = px.scatter(
                plot_data,
                x=x_col,
                y=y_col,
                color=color_arg,
                title=f"Scatter Relationship: {y_col} vs {x_col}"
            )
        elif chart_type == "Pie / Donut":
            fig = px.pie(
                plot_data,
                names=x_col,
                values=y_col,
                hole=0.4,
                title=f"{y_col} Composition by {x_col}"
            )
        elif chart_type == "Histogram":
            fig = px.histogram(
                plot_data,
                x=x_col,
                color=color_arg,
                marginal="box",
                title=f"Distribution of {x_col}"
            )
        elif chart_type == "Box Plot":
            fig = px.box(
                plot_data,
                x=x_col if not pd.api.types.is_numeric_dtype(plot_data[x_col]) else None,
                y=y_col if y_col else x_col,
                color=color_arg,
                title=f"Box Plot Spread for {y_col or x_col}"
            )
        elif chart_type == "Treemap":
            path_cols = [x_col]
            if color_arg and color_arg not in path_cols:
                path_cols.append(color_arg)
            fig = px.treemap(
                plot_data,
                path=path_cols,
                values=y_col,
                title=f"Treemap: {y_col} across {', '.join(path_cols)}"
            )
        else:
            fig = px.bar(plot_data, x=x_col, y=y_col)

        fig.update_layout(**PLOTLY_LAYOUT_DEFAULTS)
        return fig

