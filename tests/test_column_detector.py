"""
Tests for ColumnDetector.
"""

import pandas as pd
from backend.column_detector import ColumnDetector
from utils.constants import ROLE_DATE, ROLE_ID, ROLE_REVENUE, ROLE_PROFIT, ROLE_CATEGORY


def test_column_detector_roles():
    data = {
        "txn_id": ["TXN-1", "TXN-2", "TXN-3"],
        "order_date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "category_name": ["Electronics", "Fashion", "Home"],
        "gross_sales": [100.5, 200.0, 350.75],
        "net_profit": [20.0, 45.0, -10.0]
    }
    df = pd.DataFrame(data)
    detector = ColumnDetector()
    roles = detector.detect_all(df)

    assert roles["txn_id"]["role"] == ROLE_ID
    assert roles["order_date"]["role"] == ROLE_DATE
    assert roles["category_name"]["role"] == ROLE_CATEGORY
    assert roles["gross_sales"]["role"] == ROLE_REVENUE
    assert roles["net_profit"]["role"] == ROLE_PROFIT


def test_column_detector_override():
    df = pd.DataFrame({"mystery_metric": [10, 20, 30]})
    detector = ColumnDetector(overrides={"mystery_metric": ROLE_REVENUE})
    roles = detector.detect_all(df)
    assert roles["mystery_metric"]["role"] == ROLE_REVENUE
    assert roles["mystery_metric"]["is_override"] is True

