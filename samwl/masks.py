"""Mask utilities: RLE encode/decode, IoU, and bounding-box extraction.

RLE format is COCO-compatible (Fortran order, run-length on the flattened
column-major array).  The encoder is vectorised with NumPy; the decoder
rebuilds the boolean mask without a Python loop.

All public functions accept and return NumPy bool_ or uint8 arrays.
"""
from __future__ import annotations

import numpy as np


def rle_encode(mask: np.ndarray) -> dict:
    """Encode a 2-D boolean/uint8 mask to COCO RLE.

    Returns ``{"size": [H, W], "counts": [int, ...]}``.
    The first count is always the number of leading *zeros* (possibly 0),
    matching pycocotools convention so round-trips are exact.
    """
    mask = np.asarray(mask, dtype=np.uint8)
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {mask.shape}")
    h, w = mask.shape
    flat = np.asfortranarray(mask).ravel(order="F")
    n = flat.size
    if n == 0:
        return {"size": [h, w], "counts": []}
    transitions = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    starts = np.concatenate(([0], transitions))
    ends = np.concatenate((transitions, [n]))
    counts = (ends - starts).tolist()
    # If the mask starts with 1s, prepend a zero-length run of 0s.
    if flat[0] == 1:
        counts = [0] + counts
    return {"size": [h, w], "counts": counts}


def rle_decode(rle: dict) -> np.ndarray:
    """Decode a COCO RLE dict to a 2-D uint8 mask."""
    h, w = rle["size"]
    counts = rle["counts"]
    if not counts:
        return np.zeros((h, w), dtype=np.uint8)
    lengths = np.array(counts, dtype=np.int64)
    total = lengths.sum()
    expected = h * w
    if total != expected:
        raise ValueError(
            f"RLE count sum {total} != H*W={expected} for size [{h},{w}]"
        )
    flat = np.zeros(total, dtype=np.uint8)
    pos = 0
    for i, length in enumerate(lengths):
        if length > 0 and (i % 2) == 1:
            flat[pos : pos + length] = 1
        pos += length
    return flat.reshape((w, h), order="C").T  # Fortran order → (H, W)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-union of two boolean/uint8 masks."""
    a = a.astype(bool)
    b = b.astype(bool)
    intersection = int((a & b).sum())
    union = int((a | b).sum())
    return intersection / union if union > 0 else 0.0


def mask_bbox(mask: np.ndarray) -> list[int]:
    """Return [x_min, y_min, x_max, y_max] from mask nonzero extents.

    Format-independent: does not trust SAM's reported bbox.
    Returns an empty list for an all-zero mask.
    """
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return []
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
