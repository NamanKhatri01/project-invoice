"""
Image cleanup helpers for the invoice OCR pipeline.

The functions here are deliberately small and independent so that they can be
chained together or swapped out. `prepare_for_ocr` wraps the common sequence
used before handing the image to Tesseract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import cv2
import numpy as np


PathLike = Union[str, Path]


def load_image(path: PathLike) -> np.ndarray:
    """Read an image from disk into a BGR numpy array."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image at {path}")
    return image


def to_grayscale(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def denoise(image: np.ndarray) -> np.ndarray:
    """Remove speckle noise while keeping the edges of characters sharp."""
    return cv2.medianBlur(image, 3)


def binarize(image: np.ndarray) -> np.ndarray:
    """
    Turn the grayscale image into pure black and white so the OCR engine has
    an easier time. Otsu picks the threshold automatically.
    """
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def deskew(image: np.ndarray) -> np.ndarray:
    """
    Best effort rotation correction. If the document is already straight this
    is close to a no-op, so it is safe to keep in the default pipeline.
    """
    coords = np.column_stack(np.where(image < 255))
    if coords.size == 0:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    if abs(angle) < 0.5:
        return image
    (h, w) = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(
        image, matrix, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def add_border(image: np.ndarray, pixels: int = 20) -> np.ndarray:
    """
    Pad the image with a white border. Tesseract tends to clip characters that
    sit right up against the edge, so a small margin noticeably improves
    accuracy on documents that were rendered without one.
    """
    return cv2.copyMakeBorder(
        image, pixels, pixels, pixels, pixels,
        cv2.BORDER_CONSTANT, value=255,
    )


def prepare_for_ocr(path: PathLike) -> np.ndarray:
    """
    Standard cleanup used before Tesseract.

    For scans that are already crisp (like the synthetic samples in this
    project) a light touch works better than an aggressive threshold, because
    Otsu can eat thin strokes such as decimal points. We therefore only apply
    a very light denoise step and rely on Tesseract's own binarization.
    """
    raw = load_image(path)
    gray = to_grayscale(raw)
    clean = cv2.GaussianBlur(gray, (3, 3), 0)
    padded = add_border(clean, pixels=20)
    return padded
