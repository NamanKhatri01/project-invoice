"""
Builds a folder of synthetic invoice PNGs together with a ground truth CSV.

The images are intentionally simple so that Tesseract can read them reliably,
but the field layout is close enough to a real invoice for the parser and the
downstream anomaly checks to be exercised properly. A small number of the
invoices are seeded with problems (huge totals, duplicated invoice numbers,
future dates, math that does not add up) so that the detector has something
interesting to flag.
"""

from __future__ import annotations

import csv
import os
import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import List

from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
IMAGES_DIR = PROJECT_ROOT / "data" / "raw_invoices"
GROUND_TRUTH = PROJECT_ROOT / "data" / "ground_truth.csv"

VENDORS = [
    "Acme Office Supplies",
    "Blue River Logistics",
    "Cypress Software Ltd",
    "Delta Industrial Parts",
    "Evergreen Catering Co",
]

PRODUCTS = [
    ("A4 Paper Ream", 4.50, 12.00),
    ("Ballpoint Pen Box", 3.20, 9.80),
    ("Toner Cartridge", 45.00, 120.00),
    ("USB Hub", 15.00, 40.00),
    ("Desk Lamp", 22.00, 65.00),
    ("Ergonomic Chair", 120.00, 320.00),
    ("Whiteboard Marker", 1.20, 4.50),
    ("Coffee Beans 1kg", 18.00, 35.00),
]


@dataclass
class LineItem:
    description: str
    quantity: int
    unit_price: float

    @property
    def line_total(self) -> float:
        return round(self.quantity * self.unit_price, 2)


@dataclass
class Invoice:
    invoice_number: str
    vendor: str
    invoice_date: date
    line_items: List[LineItem] = field(default_factory=list)
    grand_total_override: float | None = None

    @property
    def grand_total(self) -> float:
        if self.grand_total_override is not None:
            return round(self.grand_total_override, 2)
        return round(sum(item.line_total for item in self.line_items), 2)


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    """Try to load a common system font, fall back to PIL's default."""
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/consola.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _render(invoice: Invoice, out_path: Path) -> None:
    width, height = 1000, 1300
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    title_font = _load_font(36)
    header_font = _load_font(22)
    body_font = _load_font(20)

    draw.text((40, 40), invoice.vendor, fill="black", font=title_font)
    draw.text((40, 100), "INVOICE", fill="black", font=header_font)

    draw.text((40, 160), f"Invoice Number: {invoice.invoice_number}", fill="black", font=body_font)
    draw.text((40, 195), f"Invoice Date: {invoice.invoice_date.strftime('%Y-%m-%d')}", fill="black", font=body_font)

    y = 270
    draw.text((40, y), "Description", fill="black", font=header_font)
    draw.text((500, y), "Qty", fill="black", font=header_font)
    draw.text((620, y), "Unit Price", fill="black", font=header_font)
    draw.text((820, y), "Line Total", fill="black", font=header_font)
    draw.line([(40, y + 32), (960, y + 32)], fill="black", width=2)

    y += 50
    for item in invoice.line_items:
        draw.text((40, y), item.description, fill="black", font=body_font)
        draw.text((500, y), str(item.quantity), fill="black", font=body_font)
        draw.text((620, y), f"{item.unit_price:.2f}", fill="black", font=body_font)
        draw.text((820, y), f"{item.line_total:.2f}", fill="black", font=body_font)
        y += 35

    y += 20
    draw.line([(500, y), (960, y)], fill="black", width=2)
    y += 15
    draw.text((620, y), "Grand Total:", fill="black", font=header_font)
    draw.text((820, y), f"{invoice.grand_total:.2f}", fill="black", font=header_font)

    img.save(out_path, "PNG")


def _make_normal_invoice(idx: int, rng: random.Random) -> Invoice:
    vendor = rng.choice(VENDORS)
    number = f"INV-{10000 + idx}"
    days_back = rng.randint(1, 80)
    invoice_date = date.today() - timedelta(days=days_back)
    items = []
    for _ in range(rng.randint(2, 5)):
        desc, low, high = rng.choice(PRODUCTS)
        qty = rng.randint(1, 8)
        price = round(rng.uniform(low, high), 2)
        items.append(LineItem(desc, qty, price))
    return Invoice(number, vendor, invoice_date, items)


def _inject_anomalies(invoices: List[Invoice], rng: random.Random) -> None:
    """Seed a handful of problems so the detector has something to catch."""
    if len(invoices) < 6:
        return

    huge = invoices[2]
    huge.line_items.append(LineItem("Emergency Server Rack", 1, 12500.00))

    dup_source = invoices[5]
    dup_copy = Invoice(
        invoice_number=dup_source.invoice_number,
        vendor=dup_source.vendor,
        invoice_date=dup_source.invoice_date,
        line_items=list(dup_source.line_items),
    )
    invoices.append(dup_copy)

    future = invoices[7 % len(invoices)]
    future.invoice_date = date.today() + timedelta(days=15)

    bad_math = invoices[9 % len(invoices)]
    real_total = sum(item.line_total for item in bad_math.line_items)
    bad_math.grand_total_override = round(real_total + 250.00, 2)

    stale = invoices[11 % len(invoices)]
    stale.invoice_date = date.today() - timedelta(days=400)


def generate(n: int = 50, seed: int = 7) -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    invoices = [_make_normal_invoice(i, rng) for i in range(n)]
    _inject_anomalies(invoices, rng)

    rows = []
    for i, inv in enumerate(invoices):
        filename = f"invoice_{i:03d}.png"
        _render(inv, IMAGES_DIR / filename)
        rows.append(
            {
                "file": filename,
                "invoice_number": inv.invoice_number,
                "vendor": inv.vendor,
                "invoice_date": inv.invoice_date.isoformat(),
                "grand_total": f"{inv.grand_total:.2f}",
                "num_line_items": len(inv.line_items),
            }
        )

    with GROUND_TRUTH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} invoices to {IMAGES_DIR}")
    print(f"Ground truth saved to {GROUND_TRUTH}")


if __name__ == "__main__":
    generate()
