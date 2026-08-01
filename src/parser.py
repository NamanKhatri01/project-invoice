"""
Turn raw OCR text into structured invoice records.

The parser leans on regular expressions with a few small heuristics for the
things regex is bad at (vendor name, table rows). Every extracted invoice is
returned as a dictionary with a nested list of line items so the downstream
code does not have to reparse anything.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from typing import List, Optional

import pandas as pd
from dateutil import parser as dateparser


INVOICE_NUMBER_RE = re.compile(r"invoice\s*(?:number|no\.?|#)\s*[:\-]?\s*([A-Z0-9\-]+)", re.IGNORECASE)
INVOICE_NUMBER_FALLBACK_RE = re.compile(r"\bINV[-\s]?\d{3,}\b", re.IGNORECASE)
DATE_LINE_RE = re.compile(r"invoice\s*date\s*[:\-]?\s*([0-9A-Za-z/\-\.\s,]+)", re.IGNORECASE)
GRAND_TOTAL_RE = re.compile(r"grand\s*total\s*[:\-]?\s*\$?\s*([0-9]+(?:[.,][0-9]{2}))", re.IGNORECASE)
MONEY_RE = re.compile(r"[0-9]+(?:\.[0-9]{2})")
LINE_ITEM_RE = re.compile(
    r"^(?P<desc>.+?)\s+(?P<qty>\d{1,3})\s+(?P<price>\d+\.\d{2})\s+(?P<total>\d+\.\d{2})\s*$"
)


@dataclass
class LineItem:
    description: str
    quantity: int
    unit_price: float
    line_total: float


@dataclass
class ParsedInvoice:
    source_file: str
    invoice_number: Optional[str] = None
    vendor: Optional[str] = None
    invoice_date: Optional[date] = None
    grand_total: Optional[float] = None
    line_items: List[LineItem] = field(default_factory=list)
    raw_text: str = ""


def _clean_lines(text: str) -> List[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def _extract_invoice_number(text: str) -> Optional[str]:
    match = INVOICE_NUMBER_RE.search(text)
    if match:
        return match.group(1).strip().upper()
    fallback = INVOICE_NUMBER_FALLBACK_RE.search(text)
    if fallback:
        return fallback.group(0).replace(" ", "").upper()
    return None


def _extract_date(text: str) -> Optional[date]:
    match = DATE_LINE_RE.search(text)
    candidates: List[str] = []
    if match:
        candidates.append(match.group(1).strip().split("\n")[0])
    # Also try any obvious yyyy-mm-dd or dd/mm/yyyy string anywhere in the text.
    candidates.extend(re.findall(r"\d{4}-\d{2}-\d{2}", text))
    candidates.extend(re.findall(r"\d{1,2}/\d{1,2}/\d{2,4}", text))
    for candidate in candidates:
        try:
            return dateparser.parse(candidate, dayfirst=False, fuzzy=True).date()
        except (ValueError, TypeError, OverflowError):
            continue
    return None


def _extract_vendor(lines: List[str]) -> Optional[str]:
    """
    The vendor is printed at the top of every invoice, above the word INVOICE.
    We pick the first non-empty line that is not obviously boilerplate.
    """
    skip_tokens = {"invoice", "invoice number", "invoice date", "bill to"}
    for line in lines[:5]:
        lowered = line.lower().strip(": ")
        if lowered in skip_tokens:
            continue
        if lowered.startswith("invoice"):
            continue
        # Vendor names have letters, boilerplate rows tend to be mostly numbers.
        letters = sum(ch.isalpha() for ch in line)
        if letters >= 3:
            return line
    return None


def _extract_grand_total(text: str) -> Optional[float]:
    match = GRAND_TOTAL_RE.search(text)
    if match:
        return float(match.group(1).replace(",", ""))
    # Fallback: the largest money-looking number on the page.
    numbers = [float(n) for n in MONEY_RE.findall(text)]
    if numbers:
        return max(numbers)
    return None


def _extract_line_items(lines: List[str]) -> List[LineItem]:
    items: List[LineItem] = []
    for line in lines:
        match = LINE_ITEM_RE.match(line)
        if not match:
            continue
        desc = match.group("desc").strip()
        if desc.lower().startswith(("grand total", "description")):
            continue
        try:
            qty = int(match.group("qty"))
            price = float(match.group("price"))
            total = float(match.group("total"))
        except ValueError:
            continue
        items.append(LineItem(desc, qty, price, total))
    return items


def parse_invoice(text: str, source_file: str) -> ParsedInvoice:
    lines = _clean_lines(text)
    invoice = ParsedInvoice(source_file=source_file, raw_text=text)
    invoice.invoice_number = _extract_invoice_number(text)
    invoice.invoice_date = _extract_date(text)
    invoice.vendor = _extract_vendor(lines)
    invoice.grand_total = _extract_grand_total(text)
    invoice.line_items = _extract_line_items(lines)
    return invoice


def to_dataframe(invoices: List[ParsedInvoice]) -> pd.DataFrame:
    """Flatten parsed invoices to one row each, keeping line items as a list."""
    rows = []
    for inv in invoices:
        row = {
            "source_file": inv.source_file,
            "invoice_number": inv.invoice_number,
            "vendor": inv.vendor,
            "invoice_date": inv.invoice_date,
            "grand_total": inv.grand_total,
            "line_item_count": len(inv.line_items),
            "line_items": [asdict(li) for li in inv.line_items],
        }
        if inv.line_items:
            prices = [li.unit_price for li in inv.line_items]
            row["avg_unit_price"] = sum(prices) / len(prices)
            row["max_unit_price"] = max(prices)
            row["sum_line_totals"] = round(sum(li.line_total for li in inv.line_items), 2)
        else:
            row["avg_unit_price"] = 0.0
            row["max_unit_price"] = 0.0
            row["sum_line_totals"] = 0.0
        rows.append(row)
    return pd.DataFrame(rows)
