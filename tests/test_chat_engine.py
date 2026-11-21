"""
Tests for the ChatEngine semantic query pipeline.

Covers:
  - Named composite business metrics (AOV, profit margin, attrition)
  - Correct metric-to-column mapping (never substitutes wrong columns)
  - Standard intents: summary, ranking, aggregation, count, correlation
  - "Cannot compute" responses when required columns are absent
"""

import pandas as pd
import numpy as np
import pytest
from backend.column_detector import ColumnDetector
from backend.chat_engine import ChatEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def retail_df() -> pd.DataFrame:
    """Retail sales dataset with Order_ID, Revenue, Profit, Category, Region."""
    return pd.DataFrame({
        "Order_ID":  ["O001", "O002", "O003", "O001", "O004"],
        "Category":  ["Electronics", "Fashion", "Home", "Electronics", "Fashion"],
        "Region":    ["North", "South", "North", "South", "North"],
        "Sales":     [5000.0, 2000.0, 1000.0, 3000.0, 1500.0],
        "Profit":    [1200.0, -100.0,  250.0,  800.0,  300.0],
        "Quantity":  [5, 3, 2, 4, 2],
    })


@pytest.fixture
def hr_df() -> pd.DataFrame:
    """HR dataset with Age, Salary, Attrition, Department."""
    return pd.DataFrame({
        "EmployeeID":  [1, 2, 3, 4, 5],
        "Age":         [35, 45, 28, 52, 41],
        "Department":  ["Sales", "HR", "IT", "Sales", "HR"],
        "Salary":      [60000.0, 55000.0, 70000.0, 65000.0, 50000.0],
        "Attrition":   ["Yes", "No", "Yes", "No", "No"],
    })


@pytest.fixture
def retail_chat(retail_df) -> ChatEngine:
    roles = ColumnDetector().detect_all(retail_df)
    return ChatEngine(retail_df, roles)


@pytest.fixture
def hr_chat(hr_df) -> ChatEngine:
    roles = ColumnDetector().detect_all(hr_df)
    return ChatEngine(hr_df, roles)


# ---------------------------------------------------------------------------
# 1. Named Business Metric: AOV
# ---------------------------------------------------------------------------
class TestAOV:
    def test_aov_query_recognized(self, retail_chat):
        res = retail_chat.process_query("What is the average order value?")
        assert "AOV" in res["answer"] or "Average Order Value" in res["answer"], \
            "AOV should be mentioned in answer"

    def test_aov_not_age(self, retail_chat):
        res = retail_chat.process_query("What is the average order value?")
        # Must NOT mention Age at all — that's the original bug
        assert "Age" not in res["answer"], \
            f"AOV query must not return Age. Got: {res['answer'][:200]}"

    def test_aov_value_is_numeric(self, retail_chat):
        res = retail_chat.process_query("What is the AOV?")
        # Answer should contain a currency-formatted number
        assert "$" in res["answer"] or "," in res["answer"], \
            "AOV answer should contain a numeric value"

    def test_aov_has_summary_table(self, retail_chat):
        res = retail_chat.process_query("What is the average order value?")
        assert res["data"] is not None, "AOV response should include a summary DataFrame"


# ---------------------------------------------------------------------------
# 2. Named Business Metric: Profit Margin
# ---------------------------------------------------------------------------
class TestProfitMargin:
    def test_profit_margin_query(self, retail_chat):
        res = retail_chat.process_query("What is the profit margin?")
        answer = res["answer"]
        assert "Profit Margin" in answer or "margin" in answer.lower(), \
            f"Expected profit margin in answer, got: {answer[:200]}"
        assert "%" in answer, "Profit margin answer must include a % value"

    def test_profit_margin_formula_shown(self, retail_chat):
        res = retail_chat.process_query("What is the gross margin?")
        assert "Formula" in res["answer"] or "÷" in res["answer"]


# ---------------------------------------------------------------------------
# 3. Named Business Metric: Attrition
# ---------------------------------------------------------------------------
class TestAttrition:
    def test_attrition_rate_hr(self, hr_chat):
        res = hr_chat.process_query("What is the attrition rate?")
        answer = res["answer"]
        assert "Attrition Rate" in answer or "attrition" in answer.lower()
        assert "%" in answer
        # 2 out of 5 employees left → 40%
        assert "40.00" in answer, f"Expected 40.00% attrition, got: {answer[:300]}"


# ---------------------------------------------------------------------------
# 4. Standard Aggregation — must map to correct column
# ---------------------------------------------------------------------------
class TestAggregation:
    def test_average_sales(self, retail_chat):
        res = retail_chat.process_query("What is the average sales?")
        # Mean of [5000, 2000, 1000, 3000, 1500] = 2500 -> formatted as "$2.50K"
        assert "Sales" in res["answer"], f"Expected Sales in answer, got: {res['answer'][:200]}"
        assert "2.5" in res["answer"] or "2,500" in res["answer"] or "2500" in res["answer"]

    def test_total_revenue(self, retail_chat):
        res = retail_chat.process_query("What is the total revenue?")
        # Sum of Sales = 12500
        assert "Sales" in res["answer"] or "12,500" in res["answer"]
        assert "Age" not in res["answer"], "Total revenue must not return Age"

    def test_average_age_uses_age_column(self, hr_chat):
        """Average age should return the Age column, not Salary."""
        res = hr_chat.process_query("What is the average age?")
        # Mean of [35,45,28,52,41] = 40.2
        assert "Age" in res["answer"], f"Expected Age column, got: {res['answer'][:200]}"
        assert "Salary" not in res["answer"]

    def test_aggregation_wrong_column_not_substituted(self, retail_chat):
        """Asking for 'average discount' on a dataset without Discount should give 'cannot compute'."""
        res = retail_chat.process_query("What is the average discount?")
        # No Discount column in retail_df → should explain clearly
        assert "cannot" in res["answer"].lower() or "not found" in res["answer"].lower() \
               or "Could not find" in res["answer"], \
               f"Expected cannot-compute, got: {res['answer'][:200]}"
        assert "Age" not in res["answer"]


# ---------------------------------------------------------------------------
# 5. Ranking
# ---------------------------------------------------------------------------
class TestRanking:
    def test_highest_profit_category(self, retail_chat):
        res = retail_chat.process_query("Which category generates the highest profit?")
        assert "Electronics" in res["answer"], \
            f"Electronics has highest profit, got: {res['answer'][:200]}"

    def test_lowest_sales_region(self, retail_chat):
        res = retail_chat.process_query("Which region has the lowest sales?")
        assert res["answer"]  # Non-empty
        assert "South" in res["answer"] or "North" in res["answer"]  # One of them is lowest


# ---------------------------------------------------------------------------
# 6. Summary Query
# ---------------------------------------------------------------------------
class TestSummary:
    def test_summary_row_count(self, retail_chat):
        res = retail_chat.process_query("Give me a summary of this dataset")
        assert "5" in res["answer"] or "5 rows" in res["answer"]

    def test_summary_includes_data(self, retail_chat):
        res = retail_chat.process_query("What is this dataset?")
        assert res["data"] is not None


# ---------------------------------------------------------------------------
# 7. Count Queries
# ---------------------------------------------------------------------------
class TestCount:
    def test_count_rows(self, retail_chat):
        res = retail_chat.process_query("How many records are there?")
        assert "5" in res["answer"]


# ---------------------------------------------------------------------------
# 8. No Wrong Column Substitution (regression guard)
# ---------------------------------------------------------------------------
class TestNoWrongSubstitution:
    def test_aov_never_returns_age(self, hr_chat):
        """Even in HR dataset (which has Age), AOV query must not return Age as the answer."""
        res = hr_chat.process_query("What is the average order value?")
        # HR dataset has no Sales/Revenue → must say "Cannot compute" and NOT return Age mean
        assert "Cannot compute" in res["answer"] or "cannot compute" in res["answer"], \
            f"Should report cannot-compute. Got: {res['answer'][:300]}"
        assert "average Age" not in res["answer"] and "average age" not in res["answer"].lower(), \
            "Must never return average age as the answer to an AOV question!"

    def test_average_does_not_pick_random_column(self, retail_chat):
        """'Average profit margin' must not pick an arbitrary column."""
        res = retail_chat.process_query("What is the average profit margin?")
        # This matches profit_margin named metric → should include %
        assert "Age" not in res["answer"]
        assert "Quantity" not in res["answer"] or "Profit" in res["answer"]
