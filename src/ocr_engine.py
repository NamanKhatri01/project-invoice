"""
Thin wrapper around pytesseract.

The rest of the pipeline only ever calls `extract_text`. Keeping the Tesseract
specifics in one place makes it easy to swap the backend later (for EasyOCR or
a cloud service) without touching the parser.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union

import pytesseract

from preprocessing import prepare_for_ocr


PathLike = Union[str, Path]


def _configure_tesseract() -> None:
    """Point pytesseract at the Windows binary when it is not on PATH."""
    override = os.environ.get("TESSERACT_CMD")
    if override:
        pytesseract.pytesseract.tesseract_cmd = override
        return
    default_windows = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.name == "nt" and os.path.exists(default_windows):
        pytesseract.pytesseract.tesseract_cmd = default_windows


_configure_tesseract()


def extract_text(image_path: PathLike, psm: int = 6) -> str:
    """
    Return the raw OCR text for a single invoice image.

    `psm=6` treats the page as a single uniform block of text which matches
    the layout of the synthetic invoices well. For messier documents you may
    want to experiment with 4 or 11.
    """
    prepared = prepare_for_ocr(image_path)
    config = f"--oem 3 --psm {psm}"
    return pytesseract.image_to_string(prepared, config=config)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ocr_engine.py <image_path>")
        sys.exit(1)
    print(extract_text(sys.argv[1]))
