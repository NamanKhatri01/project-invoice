"""
Thin wrapper around the pipeline for one-off invoice uploads.

The batch runner in `run_pipeline.py` works on a folder of images at once,
because the multivariate anomaly signals (Isolation Forest, per vendor
z-score, cross-invoice duplicate detection) only make sense across a
dataset. For the web app we also need to be able to process a single image
and return everything the UI needs, so this module wraps that path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd

from ocr_engine import extract_text
from parser import parse_invoice, to_dataframe
from validator import validate


def process_single(image_path: str | Path) -> Dict[str, Any]:
    """
    Run OCR, parsing and rule based validation on a single invoice image.

    Returns a dictionary shaped for template rendering: the extracted fields,
    the list of line items, a boolean `is_anomaly` and the list of reasons.
    Multivariate anomaly detection is intentionally skipped here because it
    requires context from the wider dataset.
    """
    image_path = Path(image_path)
    raw_text = extract_text(image_path)
    parsed = parse_invoice(raw_text, image_path.name)
    df = to_dataframe([parsed])
    df = validate(df)

    row = df.iloc[0]
    reasons = list(row.get("rule_reasons") or [])
    return {
        "source_file": row["source_file"],
        "invoice_number": row["invoice_number"],
        "vendor": row["vendor"],
        "invoice_date": str(row["invoice_date"]) if row["invoice_date"] is not None else None,
        "grand_total": row["grand_total"],
        "line_item_count": int(row["line_item_count"]),
        "line_items": row["line_items"],
        "sum_line_totals": row["sum_line_totals"],
        "is_anomaly": len(reasons) > 0,
        "anomaly_reason": "; ".join(reasons),
        "raw_text": raw_text,
    }
