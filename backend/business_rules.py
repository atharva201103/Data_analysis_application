"""
Business Rules Engine (Layer 2) for DataSense AI.
Evaluates deterministic business heuristics and patterns against statistical facts.
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from utils.constants import (
    ROLE_REVENUE, ROLE_PROFIT, ROLE_COST, ROLE_QUANTITY,
    ROLE_CATEGORY, ROLE_REGION, ROLE_DATE, ROLE_CUSTOMER,
    ROLE_STATUS
)
from utils.helpers import format_currency, format_percentage, format_number


class BusinessRuleEngine:
    """Evaluates business rules over calculated analytics."""

    @staticmethod
    def evaluate_rules(
        df: pd.DataFrame,
        column_roles: Dict[str, Dict[str, Any]],
        domain: str = "general"
    ) -> List[Dict[str, Any]]:
        """
        Evaluate business rules across available dimensions and metrics.
        Returns a list of structured findings.
        """
        findings = []

        rev_col = BusinessRuleEngine._find_col(column_roles, ROLE_REVENUE, df)
        profit_col = BusinessRuleEngine._find_col(column_roles, ROLE_PROFIT, df)
        cost_col = BusinessRuleEngine._find_col(column_roles, ROLE_COST, df)
        cat_col = BusinessRuleEngine._find_col(column_roles, ROLE_CATEGORY, df)
        region_col = BusinessRuleEngine._find_col(column_roles, ROLE_REGION, df)
        date_col = BusinessRuleEngine._find_col(column_roles, ROLE_DATE, df)
        cust_col = BusinessRuleEngine._find_col(column_roles, ROLE_CUSTOMER, df)
        status_col = BusinessRuleEngine._find_col(column_roles, ROLE_STATUS, df)

        # -------------------------------------------------------------
        # Rule 1: Category Pareto & Top/Bottom Disparity
        # -------------------------------------------------------------
        if cat_col and (rev_col or profit_col):
            metric = rev_col or profit_col
            valid_df = df[[cat_col, metric]].dropna()
            if len(valid_df) > 0 and valid_df[cat_col].nunique() >= 2:
                cat_summary = (
                    valid_df.groupby(cat_col)[metric]
                    .sum()
                    .reset_index()
                    .sort_values(by=metric, ascending=False)
                )
                total_val = cat_summary[metric].sum()
                if total_val > 0:
                    top_cat = cat_summary.iloc[0]
                    bottom_cat = cat_summary.iloc[-1]
                    top_share = (top_cat[metric] / total_val) * 100
                    bottom_share = (bottom_cat[metric] / total_val) * 100

                    # Pareto check: do top 20% categories account for > 60% of volume?
                    n_cats = len(cat_summary)
                    top_20_count = max(1, int(np.ceil(0.2 * n_cats)))
                    top_20_sum = cat_summary.head(top_20_count)[metric].sum()
                    pareto_share = (top_20_sum / total_val) * 100

                    findings.append({
                        "rule_id": "top_category_dominance",
                        "title": f"Primary Revenue Driver: '{top_cat[cat_col]}'",
                        "category": "top_performer",
                        "observation": (
                            f"The '{top_cat[cat_col]}' category is the top performing segment, generating "
                            f"{format_currency(top_cat[metric])} ({top_share:.1f}% of total {metric}). "
                            f"In contrast, the lowest performing segment '{bottom_cat[cat_col]}' generated only "
                            f"{format_currency(bottom_cat[metric])} ({bottom_share:.1f}%)."
                        ),
                        "evidence": f"Top segment: {format_currency(top_cat[metric])} vs Bottom segment: {format_currency(bottom_cat[metric])}. Total {metric}: {format_currency(total_val)}.",
                        "metrics": {
                            "top_category": str(top_cat[cat_col]),
                            "top_value": float(top_cat[metric]),
                            "top_share_pct": round(top_share, 2),
                            "bottom_category": str(bottom_cat[cat_col]),
                            "bottom_value": float(bottom_cat[metric]),
                            "bottom_share_pct": round(bottom_share, 2),
                            "pareto_share_pct": round(pareto_share, 2)
                        }
                    })

                    if pareto_share >= 60.0 and n_cats >= 3:
                        findings.append({
                            "rule_id": "pareto_concentration_risk",
                            "title": "Category Revenue Concentration (Pareto Asymmetry)",
                            "category": "risk",
                            "observation": (
                                f"The top {top_20_count} category segment(s) account for {pareto_share:.1f}% "
                                f"of the portfolio's total {metric}, indicating high category dependency."
                            ),
                            "evidence": f"Top {top_20_count}/{n_cats} categories contribute {pareto_share:.1f}% of cumulative {metric}.",
                            "metrics": {
                                "top_count": top_20_count,
                                "total_categories": n_cats,
                                "concentration_pct": round(pareto_share, 2)
                            }
                        })

        # -------------------------------------------------------------
        # Rule 2: Profitability Leakage / Margin Disparities
        # -------------------------------------------------------------
        if rev_col and profit_col and cat_col:
            valid_df = df[[cat_col, rev_col, profit_col]].dropna()
            if len(valid_df) > 0 and valid_df[cat_col].nunique() >= 2:
                grouped = (
                    valid_df.groupby(cat_col)
                    .agg({rev_col: "sum", profit_col: "sum"})
                    .reset_index()
                )
                grouped["margin_pct"] = (grouped[profit_col] / grouped[rev_col].replace(0, np.nan)) * 100

                # Check for negative margin or margin diluters
                loss_makers = grouped[grouped["margin_pct"] < 0]
                if len(loss_makers) > 0:
                    worst_loss = loss_makers.sort_values(by=profit_col).iloc[0]
                    findings.append({
                        "rule_id": "negative_margin_leakage",
                        "title": f"Profit Margin Erosion in '{worst_loss[cat_col]}'",
                        "category": "risk",
                        "observation": (
                            f"Segment '{worst_loss[cat_col]}' is operating at a net loss with "
                            f"{format_currency(worst_loss[profit_col])} profit on {format_currency(worst_loss[rev_col])} "
                            f"revenue (Margin: {worst_loss['margin_pct']:.1f}%)."
                        ),
                        "evidence": f"Loss incurred: {format_currency(worst_loss[profit_col])} on sales of {format_currency(worst_loss[rev_col])}.",
                        "metrics": {
                            "segment": str(worst_loss[cat_col]),
                            "revenue": float(worst_loss[rev_col]),
                            "profit": float(worst_loss[profit_col]),
                            "margin_pct": round(float(worst_loss["margin_pct"]), 2)
                        }
                    })

                # Highest margin leader
                profitable = grouped[grouped["margin_pct"] > 0]
                if len(profitable) > 0:
                    best_margin = profitable.sort_values(by="margin_pct", ascending=False).iloc[0]
                    findings.append({
                        "rule_id": "high_margin_champion",
                        "title": f"Margin Efficiency Leader: '{best_margin[cat_col]}'",
                        "category": "opportunity",
                        "observation": (
                            f"'{best_margin[cat_col]}' achieves the highest profit efficiency with a "
                            f"{best_margin['margin_pct']:.1f}% margin, generating {format_currency(best_margin[profit_col])} "
                            f"profit from {format_currency(best_margin[rev_col])} revenue."
                        ),
                        "evidence": f"Top margin: {best_margin['margin_pct']:.1f}% vs Portfolio average: {(grouped[profit_col].sum() / grouped[rev_col].sum() * 100):.1f}%.",
                        "metrics": {
                            "segment": str(best_margin[cat_col]),
                            "margin_pct": round(float(best_margin["margin_pct"]), 2),
                            "profit": float(best_margin[profit_col])
                        }
                    })

        # -------------------------------------------------------------
        # Rule 3: Regional Disparity
        # -------------------------------------------------------------
        if region_col and rev_col:
            valid_reg = df[[region_col, rev_col]].dropna()
            if len(valid_reg) > 0 and valid_reg[region_col].nunique() >= 2:
                reg_summary = (
                    valid_reg.groupby(region_col)[rev_col]
                    .sum()
                    .reset_index()
                    .sort_values(by=rev_col, ascending=False)
                )
                top_reg = reg_summary.iloc[0]
                bottom_reg = reg_summary.iloc[-1]
                reg_total = reg_summary[rev_col].sum()

                findings.append({
                    "rule_id": "regional_performance_variance",
                    "title": f"Regional Variance: '{top_reg[region_col]}' Leads Performance",
                    "category": "general",
                    "observation": (
                        f"The '{top_reg[region_col]}' territory is the leading contributor, delivering "
                        f"{format_currency(top_reg[rev_col])} ({top_reg[rev_col]/reg_total*100:.1f}% of total). "
                        f"The lowest volume territory '{bottom_reg[region_col]}' generated {format_currency(bottom_reg[rev_col])} "
                        f"({bottom_reg[rev_col]/reg_total*100:.1f}%)."
                    ),
                    "evidence": f"Leading region delivered {format_currency(top_reg[rev_col])}; lagging region delivered {format_currency(bottom_reg[rev_col])}.",
                    "metrics": {
                        "top_region": str(top_reg[region_col]),
                        "top_revenue": float(top_reg[rev_col]),
                        "bottom_region": str(bottom_reg[region_col]),
                        "bottom_revenue": float(bottom_reg[rev_col])
                    }
                })

        # -------------------------------------------------------------
        # Rule 4: Chronological / Trend Velocity
        # -------------------------------------------------------------
        if date_col and rev_col:
            try:
                temp_time = df[[date_col, rev_col]].dropna().copy()
                temp_time[date_col] = pd.to_datetime(temp_time[date_col], errors="coerce")
                temp_time = temp_time.dropna().sort_values(by=date_col)

                if len(temp_time) > 10:
                    # Split into first half vs second half
                    mid_idx = len(temp_time) // 2
                    first_half_sum = temp_time.iloc[:mid_idx][rev_col].sum()
                    second_half_sum = temp_time.iloc[mid_idx:][rev_col].sum()

                    if first_half_sum > 0:
                        velocity_pct = ((second_half_sum - first_half_sum) / first_half_sum) * 100
                        direction = "expansion" if velocity_pct > 0 else "contraction"
                        trend_category = "opportunity" if velocity_pct > 5 else ("warning" if velocity_pct < -5 else "general")

                        findings.append({
                            "rule_id": "timeline_growth_velocity",
                            "title": f"Timeline Revenue Velocity: {abs(velocity_pct):.1f}% {direction.capitalize()}",
                            "category": trend_category,
                            "observation": (
                                f"Revenue experienced a {abs(velocity_pct):.1f}% {direction} between the earlier half "
                                f"({format_currency(first_half_sum)}) and latter half ({format_currency(second_half_sum)}) "
                                f"of the recorded timeline."
                            ),
                            "evidence": f"Period 1: {format_currency(first_half_sum)} -> Period 2: {format_currency(second_half_sum)} (Delta: {velocity_pct:+.1f}%).",
                            "metrics": {
                                "period_1_sum": float(first_half_sum),
                                "period_2_sum": float(second_half_sum),
                                "growth_rate_pct": round(velocity_pct, 2)
                            }
                        })
            except Exception:
                pass

        # -------------------------------------------------------------
        # Rule 5: HR Analytics Rules (Attrition Drivers)
        # -------------------------------------------------------------
        if domain == "hr_analytics":
            # Check attrition vs overtime
            if "Overtime" in df.columns and "Attrition" in df.columns:
                try:
                    ot_summary = pd.crosstab(df["Overtime"], df["Attrition"], normalize="index")
                    if "Yes" in ot_summary.columns and "Yes" in ot_summary.index and "No" in ot_summary.index:
                        ot_attr_rate = ot_summary.loc["Yes", "Yes"] * 100
                        no_ot_attr_rate = ot_summary.loc["No", "Yes"] * 100

                        if ot_attr_rate > no_ot_attr_rate + 5:
                            findings.append({
                                "rule_id": "hr_overtime_attrition_risk",
                                "title": "Overtime Workload Correlates with Higher Attrition",
                                "category": "risk",
                                "observation": (
                                    f"Employees working overtime exhibit an attrition rate of {ot_attr_rate:.1f}%, "
                                    f"compared to only {no_ot_attr_rate:.1f}% among employees without overtime."
                                ),
                                "evidence": f"Overtime Attrition: {ot_attr_rate:.1f}% vs Non-Overtime: {no_ot_attr_rate:.1f}%.",
                                "metrics": {
                                    "overtime_attrition_pct": round(ot_attr_rate, 2),
                                    "standard_attrition_pct": round(no_ot_attr_rate, 2)
                                }
                            })
                except Exception:
                    pass

        return findings

    @staticmethod
    def _find_col(column_roles: Dict[str, Dict[str, Any]], target_role: str, df: pd.DataFrame) -> Optional[str]:
        for col, meta in column_roles.items():
            if meta.get("role") == target_role and col in df.columns:
                return col
        return None

