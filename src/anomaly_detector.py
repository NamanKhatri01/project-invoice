"""
Anomaly detection for the invoice dataset.

Three complementary signals are combined:

1. Per vendor z-score on the grand total. Useful for the classic case of a
   vendor whose typical bill is small suddenly submitting something much
   larger.
2. Isolation Forest over a handful of numeric features. This catches invoices
   that are unusual across several dimensions at once even when no single
   column is extreme on its own.
3. Fuzzy duplicate detection. Two invoices with the same or nearly-identical
   invoice number, vendor and amount are almost certainly a double payment.
"""

from __future__ import annotations

from datetime import date
from typing import List

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.ensemble import IsolationForest


Z_THRESHOLD = 2.5
ISO_CONTAMINATION = 0.1
DUPLICATE_SCORE = 92


def _vendor_zscores(df: pd.DataFrame) -> List[List[str]]:
    reasons: List[List[str]] = [[] for _ in range(len(df))]
    if "grand_total" not in df.columns:
        return reasons
    grouped = df.groupby("vendor")["grand_total"]
    means = grouped.transform("mean")
    stds = grouped.transform("std").replace(0, np.nan)
    zscores = (df["grand_total"] - means) / stds
    for idx, z in zscores.items():
        if pd.notna(z) and abs(z) > Z_THRESHOLD:
            reasons[df.index.get_loc(idx)].append(
                f"amount is {z:+.1f} standard deviations from this vendor's average"
            )
    return reasons


def _isolation_forest(df: pd.DataFrame, today: date) -> List[List[str]]:
    reasons: List[List[str]] = [[] for _ in range(len(df))]
    features = pd.DataFrame(index=df.index)
    features["grand_total"] = df["grand_total"].fillna(0.0)
    features["line_item_count"] = df["line_item_count"].fillna(0)
    features["avg_unit_price"] = df["avg_unit_price"].fillna(0.0)
    features["max_unit_price"] = df["max_unit_price"].fillna(0.0)

    def days_since(d):
        if d is None or pd.isna(d):
            return 0
        if isinstance(d, pd.Timestamp):
            d = d.date()
        return (today - d).days

    features["days_since_invoice"] = df["invoice_date"].apply(days_since)

    if len(features) < 5:
        return reasons

    model = IsolationForest(
        n_estimators=200,
        contamination=ISO_CONTAMINATION,
        random_state=42,
    )
    preds = model.fit_predict(features.values)
    scores = model.decision_function(features.values)
    for pos, (pred, score) in enumerate(zip(preds, scores)):
        if pred == -1:
            reasons[pos].append(
                f"isolation forest flagged this invoice as an outlier (score {score:.3f})"
            )
    return reasons


def _fuzzy_duplicates(df: pd.DataFrame) -> List[List[str]]:
    reasons: List[List[str]] = [[] for _ in range(len(df))]
    records = df.to_dict("records")
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            a, b = records[i], records[j]
            num_a = (a.get("invoice_number") or "").upper()
            num_b = (b.get("invoice_number") or "").upper()
            if not num_a or not num_b:
                continue
            similarity = fuzz.ratio(num_a, num_b)
            same_vendor = (a.get("vendor") or "").lower() == (b.get("vendor") or "").lower()
            amt_a = a.get("grand_total") or 0.0
            amt_b = b.get("grand_total") or 0.0
            amt_close = abs(amt_a - amt_b) <= max(1.0, 0.01 * max(amt_a, amt_b))
            if similarity >= DUPLICATE_SCORE and same_vendor and amt_close:
                note_a = f"possible duplicate of {b.get('source_file')} (score {similarity})"
                note_b = f"possible duplicate of {a.get('source_file')} (score {similarity})"
                reasons[i].append(note_a)
                reasons[j].append(note_b)
    return reasons


def detect(df: pd.DataFrame, today: date | None = None) -> pd.DataFrame:
    today = today or date.today()
    df = df.reset_index(drop=True).copy()

    rule_reasons = df.get("rule_reasons", pd.Series([[] for _ in range(len(df))]))
    z_reasons = _vendor_zscores(df)
    iso_reasons = _isolation_forest(df, today)
    dup_reasons = _fuzzy_duplicates(df)

    combined: List[str] = []
    flags: List[bool] = []
    for i in range(len(df)):
        merged = list(rule_reasons.iloc[i]) + z_reasons[i] + iso_reasons[i] + dup_reasons[i]
        # Deduplicate while preserving order.
        seen = set()
        unique = []
        for reason in merged:
            if reason not in seen:
                seen.add(reason)
                unique.append(reason)
        combined.append("; ".join(unique))
        flags.append(len(unique) > 0)

    df["is_anomaly"] = flags
    df["anomaly_reason"] = combined
    return df
