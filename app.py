"""
Flask front end for the invoice pipeline.

Two things happen on this page:

1. Visitors can upload a single invoice image and see the extracted fields
   plus any rule based warnings.
2. The pre-computed sample dataset (produced by `run_pipeline.py`) is
   rendered as a table underneath, with anomalous rows highlighted, so a
   reviewer can see the batch pipeline output without having to install
   anything locally.
"""

from __future__ import annotations

import ast
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request, send_file, abort

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from pipeline_api import process_single


ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MAX_UPLOAD_BYTES = 8 * 1024 * 1024

CSV_PATH = HERE / "output" / "invoices_extracted.csv"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES


def _load_sample_rows():
    """Load the pre-computed CSV so we can render it as a table."""
    if not CSV_PATH.exists():
        return []
    df = pd.read_csv(CSV_PATH)
    keep = [
        "source_file", "invoice_number", "vendor", "invoice_date",
        "grand_total", "line_item_count", "is_anomaly", "anomaly_reason",
    ]
    df = df[[c for c in keep if c in df.columns]].fillna("")
    return df.to_dict("records")


def _allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        sample_rows=_load_sample_rows(),
        result=None,
        error=None,
    )


@app.route("/upload", methods=["POST"])
def upload():
    error = None
    result = None

    uploaded = request.files.get("invoice")
    if uploaded is None or uploaded.filename == "":
        error = "Please pick an invoice image before uploading."
    elif not _allowed(uploaded.filename):
        error = f"Only these file types are supported: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
    else:
        suffix = Path(uploaded.filename).suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            uploaded.save(tmp.name)
            tmp_path = tmp.name
        try:
            result = process_single(tmp_path)
            result["display_name"] = uploaded.filename
        except Exception as exc:
            error = f"Could not process the image: {exc}"
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    return render_template(
        "index.html",
        sample_rows=_load_sample_rows(),
        result=result,
        error=error,
    )


@app.route("/download", methods=["GET"])
def download():
    if not CSV_PATH.exists():
        abort(404, "Sample CSV has not been generated yet.")
    return send_file(
        CSV_PATH,
        as_attachment=True,
        download_name="invoices_extracted.csv",
        mimetype="text/csv",
    )


@app.template_filter("parse_items")
def parse_items(value):
    """CSV stores line items as a stringified list. Turn it back for display."""
    if not value or value == "[]":
        return []
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
