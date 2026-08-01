"""
End to end runner.

Reads every image in `data/raw_invoices/`, sends it through the OCR pipeline,
parses the text, validates the result, flags anomalies, and writes the final
CSV and the figures used in the report.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from ocr_engine import extract_text
from parser import parse_invoice, to_dataframe, ParsedInvoice
from validator import validate
from anomaly_detector import detect


PROJECT_ROOT = HERE.parent
IMAGES_DIR = PROJECT_ROOT / "data" / "raw_invoices"
GROUND_TRUTH = PROJECT_ROOT / "data" / "ground_truth.csv"
OUTPUT_DIR = PROJECT_ROOT / "output"
FIGURES_DIR = PROJECT_ROOT / "report" / "figures"


def _ocr_all() -> List[ParsedInvoice]:
    parsed: List[ParsedInvoice] = []
    files = sorted(IMAGES_DIR.glob("*.png"))
    if not files:
        raise SystemExit(
            f"No invoice images found in {IMAGES_DIR}. "
            "Run `python src/generate_samples.py` first."
        )
    for path in files:
        print(f"OCR: {path.name}")
        text = extract_text(path)
        parsed.append(parse_invoice(text, path.name))
    return parsed


def _ocr_accuracy(df: pd.DataFrame) -> dict:
    if not GROUND_TRUTH.exists():
        return {}
    truth = pd.read_csv(GROUND_TRUTH)
    merged = df.merge(truth, left_on="source_file", right_on="file", suffixes=("_pred", "_true"))
    total = len(merged)
    if total == 0:
        return {}
    num_match = (merged["invoice_number_pred"].fillna("") == merged["invoice_number_true"].fillna("")).mean()
    vendor_match = (
        merged["vendor_pred"].fillna("").str.strip().str.lower()
        == merged["vendor_true"].fillna("").str.strip().str.lower()
    ).mean()
    pred_total = pd.to_numeric(merged["grand_total_pred"], errors="coerce")
    true_total = pd.to_numeric(merged["grand_total_true"], errors="coerce")
    total_match = ((pred_total - true_total).abs() <= 0.05).mean()
    date_true = pd.to_datetime(merged["invoice_date_true"], errors="coerce").dt.date
    date_pred = pd.to_datetime(merged["invoice_date_pred"], errors="coerce").dt.date
    date_match = (date_pred == date_true).mean()
    return {
        "invoice_number_accuracy": round(float(num_match), 3),
        "vendor_accuracy": round(float(vendor_match), 3),
        "grand_total_accuracy": round(float(total_match), 3),
        "invoice_date_accuracy": round(float(date_match), 3),
        "invoices_evaluated": int(total),
    }


def _plot_figures(df: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    amounts_all = pd.to_numeric(df["grand_total"], errors="coerce")
    amounts = amounts_all.dropna()

    plt.figure(figsize=(6, 4))
    plt.boxplot(amounts, orientation="horizontal")
    plt.title("Distribution of Invoice Grand Totals")
    plt.xlabel("Grand Total")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "boxplot.png", dpi=130)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.hist(amounts, bins=15, color="#4c72b0", edgecolor="black")
    plt.title("Histogram of Invoice Grand Totals")
    plt.xlabel("Grand Total")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "histogram.png", dpi=130)
    plt.close()

    plt.figure(figsize=(6, 4))
    colours = ["#d62728" if flag else "#2ca02c" for flag in df["is_anomaly"]]
    plt.scatter(df["line_item_count"], amounts_all, c=colours, alpha=0.75, edgecolor="black")
    plt.title("Line Item Count vs Grand Total")
    plt.xlabel("Line Item Count")
    plt.ylabel("Grand Total")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "scatter.png", dpi=130)
    plt.close()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    parsed = _ocr_all()
    df = to_dataframe(parsed)
    df = validate(df)
    df = detect(df)

    export = df.drop(columns=["raw_text"], errors="ignore").copy()
    export["line_items"] = export["line_items"].apply(lambda items: str(items))
    export.drop(columns=["rule_reasons"], errors="ignore", inplace=True)
    csv_path = OUTPUT_DIR / "invoices_extracted.csv"
    export.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}")

    accuracy = _ocr_accuracy(df)
    if accuracy:
        print("\nOCR accuracy vs ground truth:")
        for key, value in accuracy.items():
            print(f"  {key}: {value}")

    _plot_figures(df)
    print(f"Figures saved to {FIGURES_DIR}")

    flagged = df[df["is_anomaly"]]
    print(f"\nFlagged {len(flagged)} of {len(df)} invoices as anomalous.")


if __name__ == "__main__":
    main()
