"""Tests for samwl.masks: RLE encode/decode, mask_iou, mask_bbox.

Verification target (Step 4):
- round-trip property: rle_decode(rle_encode(m)) == m for 200 random masks
- edge cases: all-zeros, all-ones, single pixel, single row/column
- mask_iou correctness: perfect overlap, zero overlap, partial
- mask_bbox correctness
"""
from __future__ import annotations

import numpy as np
import pytest

from samwl.masks import mask_bbox, mask_iou, rle_decode, rle_encode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)


def random_mask(h: int, w: int, density: float = 0.5) -> np.ndarray:
    return (RNG.random((h, w)) < density).astype(np.uint8)


# ---------------------------------------------------------------------------
# Round-trip property: 200 random masks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(200))
def test_rle_roundtrip(seed: int) -> None:
    """rle_decode(rle_encode(m)) == m for varied shapes and densities."""
    rng = np.random.default_rng(seed)
    h = rng.integers(1, 128)
    w = rng.integers(1, 128)
    density = rng.random()
    mask = (rng.random((h, w)) < density).astype(np.uint8)
    rt = rle_decode(rle_encode(mask))
    assert rt.shape == mask.shape, f"shape mismatch: {rt.shape} != {mask.shape}"
    assert np.array_equal(rt, mask), "round-trip mismatch"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_all_zeros() -> None:
    mask = np.zeros((10, 15), dtype=np.uint8)
    rt = rle_decode(rle_encode(mask))
    assert np.array_equal(rt, mask)


def test_all_ones() -> None:
    mask = np.ones((10, 15), dtype=np.uint8)
    rt = rle_decode(rle_encode(mask))
    assert np.array_equal(rt, mask)


def test_single_pixel() -> None:
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[2, 3] = 1
    rt = rle_decode(rle_encode(mask))
    assert np.array_equal(rt, mask)


def test_single_row() -> None:
    mask = np.zeros((1, 20), dtype=np.uint8)
    mask[0, 5:15] = 1
    rt = rle_decode(rle_encode(mask))
    assert np.array_equal(rt, mask)


def test_single_column() -> None:
    mask = np.zeros((20, 1), dtype=np.uint8)
    mask[3:17, 0] = 1
    rt = rle_decode(rle_encode(mask))
    assert np.array_equal(rt, mask)


def test_encode_2d_required() -> None:
    with pytest.raises(ValueError, match="2-D"):
        rle_encode(np.zeros((5, 5, 3), dtype=np.uint8))


def test_decode_count_mismatch() -> None:
    rle = {"size": [4, 4], "counts": [5, 5, 7]}  # sum=17, H*W=16
    with pytest.raises(ValueError, match="count sum"):
        rle_decode(rle)


# ---------------------------------------------------------------------------
# mask_iou
# ---------------------------------------------------------------------------

def test_mask_iou_perfect_overlap() -> None:
    a = np.ones((10, 10), dtype=np.uint8)
    assert mask_iou(a, a) == pytest.approx(1.0)


def test_mask_iou_zero_overlap() -> None:
    a = np.zeros((10, 10), dtype=np.uint8)
    a[:, :5] = 1
    b = np.zeros((10, 10), dtype=np.uint8)
    b[:, 5:] = 1
    assert mask_iou(a, b) == pytest.approx(0.0)


def test_mask_iou_partial() -> None:
    a = np.zeros((10, 10), dtype=np.uint8)
    a[:, :6] = 1  # 60 px
    b = np.zeros((10, 10), dtype=np.uint8)
    b[:, 4:] = 1  # 60 px, overlap at cols 4-5 = 20 px
    iou = mask_iou(a, b)
    # union = 100, intersection = 20 => 0.2
    assert iou == pytest.approx(20 / 100)


def test_mask_iou_empty_masks() -> None:
    a = np.zeros((5, 5), dtype=np.uint8)
    assert mask_iou(a, a) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# mask_bbox
# ---------------------------------------------------------------------------

def test_mask_bbox_basic() -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:5, 3:7] = 1
    bbox = mask_bbox(mask)
    assert bbox == [3, 2, 6, 4]  # x_min, y_min, x_max, y_max


def test_mask_bbox_single_pixel() -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[4, 7] = 1
    assert mask_bbox(mask) == [7, 4, 7, 4]


def test_mask_bbox_empty() -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)
    assert mask_bbox(mask) == []


def test_mask_bbox_full_image() -> None:
    mask = np.ones((8, 12), dtype=np.uint8)
    assert mask_bbox(mask) == [0, 0, 11, 7]
