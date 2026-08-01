"""
Rule based sanity checks that run before the machine learning step.

Every check appends a short human readable reason to a list stored on the
invoice row. The anomaly detector later merges these reasons with its own so
the final CSV has a single `anomaly_reason` column.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import List

import pandas as pd


STALE_DAYS = 90
MATH_TOLERANCE = 0.05


def _check_math(row: pd.Series) -> List[str]:
    reasons: List[str] = []
    items = row.get("line_items") or []
    for item in items:
        qty = item.get("quantity")
        price = item.get("unit_price")
        total = item.get("line_total")
        if qty is None or price is None or total is None:
            continue
        expected = round(qty * price, 2)
        if abs(expected - total) > MATH_TOLERANCE:
            reasons.append(
                f"line item math mismatch on '{item.get('description', '?')}' "
                f"(expected {expected:.2f}, got {total:.2f})"
            )
    grand = row.get("grand_total")
    sum_lines = row.get("sum_line_totals")
    if grand is not None and sum_lines is not None and sum_lines > 0:
        if abs(grand - sum_lines) > MATH_TOLERANCE:
            reasons.append(
                f"grand total {grand:.2f} does not match line total sum {sum_lines:.2f}"
            )
    return reasons


def _check_dates(row: pd.Series, today: date) -> List[str]:
    reasons: List[str] = []
    inv_date = row.get("invoice_date")
    if inv_date is None or pd.isna(inv_date):
        reasons.append("invoice date could not be parsed")
        return reasons
    if isinstance(inv_date, pd.Timestamp):
        inv_date = inv_date.date()
    if inv_date > today:
        reasons.append(f"invoice date {inv_date} is in the future")
    elif inv_date < today - timedelta(days=STALE_DAYS):
        age = (today - inv_date).days
        reasons.append(f"invoice date is {age} days old (older than {STALE_DAYS} day threshold)")
    return reasons


def _check_missing(row: pd.Series) -> List[str]:
    reasons: List[str] = []
    for field in ("invoice_number", "vendor", "grand_total"):
        value = row.get(field)
        if value is None or (isinstance(value, float) and pd.isna(value)):
            reasons.append(f"missing {field.replace('_', ' ')}")
    return reasons


def validate(df: pd.DataFrame, today: date | None = None) -> pd.DataFrame:
    today = today or date.today()
    reasons_col: List[List[str]] = []
    for _, row in df.iterrows():
        combined: List[str] = []
        combined.extend(_check_missing(row))
        combined.extend(_check_math(row))
        combined.extend(_check_dates(row, today))
        reasons_col.append(combined)
    df = df.copy()
    df["rule_reasons"] = reasons_col
    return df
