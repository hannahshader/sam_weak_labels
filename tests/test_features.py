"""Tests for samwl.features — Step 7 verification.

Verification criteria from the plan:
1. Feature vector has the declared length (FEATURE_DIM).
2. No NaN or inf for a degenerate 1-pixel mask.
3. Bitwise identical across two runs on the same input.
"""
from __future__ import annotations

import numpy as np
import pytest

from samwl.features import (
    DINO_DIM,
    FEATURE_DIM,
    HAND_DIM,
    HAND_FEATURE_NAMES,
    FeatureExtractor,
)
from samwl.masks import mask_bbox, rle_encode


# ── Fixture ───────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def extractor() -> FeatureExtractor:
    """Load DINOv2 once for the whole module; CPU ensures determinism."""
    return FeatureExtractor(device="cpu")


# ── Helper ────────────────────────────────────────────────────────────────────


def _make_candidate(
    mask: np.ndarray,
    pred_iou: float = 0.9,
    stability_score: float = 0.85,
) -> dict:
    """Build a candidate dict matching the samwl.candidates record schema."""
    bbox = mask_bbox(mask)
    if not bbox:
        # mask_bbox returns [] for all-zero masks; synthesise from nonzero.
        ys, xs = np.nonzero(mask)
        bbox = (
            [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
            if xs.size > 0
            else [0, 0, 0, 0]
        )
    return {
        "image_id": "test_img",
        "rung": "coarse",
        "rle": rle_encode(mask),
        "bbox": bbox,
        "area": int(mask.sum()),
        "pred_iou": pred_iou,
        "stability_score": stability_score,
    }


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_layout_constants() -> None:
    """FEATURE_DIM == DINO_DIM * 2 + HAND_DIM and HAND_FEATURE_NAMES is complete."""
    assert FEATURE_DIM == DINO_DIM * 2 + HAND_DIM
    assert len(HAND_FEATURE_NAMES) == HAND_DIM


def test_vector_length(extractor: FeatureExtractor) -> None:
    """Output shape is (1, FEATURE_DIM) for a normal mask."""
    h, w = 64, 80
    image = np.random.default_rng(0).integers(0, 256, (h, w, 3), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[20:45, 15:60] = 1
    result = extractor.extract_batch(image, [_make_candidate(mask)])
    assert result.shape == (1, FEATURE_DIM), (
        f"expected (1, {FEATURE_DIM}), got {result.shape}"
    )


def test_no_nan_inf_1px_mask(extractor: FeatureExtractor) -> None:
    """A degenerate 1-pixel mask must produce a fully-finite feature vector."""
    h, w = 30, 40
    image = np.random.default_rng(7).integers(0, 256, (h, w, 3), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[15, 20] = 1
    result = extractor.extract_batch(image, [_make_candidate(mask)])
    assert result.shape == (1, FEATURE_DIM)
    assert not np.any(np.isnan(result)), "NaN in feature vector for 1-pixel mask"
    assert not np.any(np.isinf(result)), "inf in feature vector for 1-pixel mask"


def test_bitwise_identical(extractor: FeatureExtractor) -> None:
    """Two calls on the same input return bitwise-identical arrays."""
    h, w = 48, 64
    rng = np.random.default_rng(42)
    image = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[10:35, 12:50] = 1
    cand = _make_candidate(mask)
    r1 = extractor.extract_batch(image, [cand])
    r2 = extractor.extract_batch(image, [cand])
    assert np.array_equal(r1, r2), (
        "Feature vectors differ across two runs on identical input"
    )


def test_empty_candidates_shape(extractor: FeatureExtractor) -> None:
    """Empty candidate list returns shape (0, FEATURE_DIM)."""
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    result = extractor.extract_batch(image, [])
    assert result.shape == (0, FEATURE_DIM)


def test_batch_multiple_candidates(extractor: FeatureExtractor) -> None:
    """Batch of N candidates has shape (N, FEATURE_DIM) with no NaN or inf."""
    h, w = 60, 80
    rng = np.random.default_rng(99)
    image = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    candidates = []
    for row_start, col_start in [(5, 5), (30, 10), (20, 50)]:
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[row_start : row_start + 15, col_start : col_start + 20] = 1
        candidates.append(_make_candidate(mask))
    result = extractor.extract_batch(image, candidates)
    assert result.shape == (3, FEATURE_DIM)
    assert not np.any(np.isnan(result)), "NaN in batch feature matrix"
    assert not np.any(np.isinf(result)), "inf in batch feature matrix"


def test_different_candidates_differ(extractor: FeatureExtractor) -> None:
    """Two clearly different masks produce different feature vectors."""
    h, w = 64, 64
    rng = np.random.default_rng(55)
    image = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)

    mask_a = np.zeros((h, w), dtype=np.uint8)
    mask_a[5:20, 5:20] = 1  # top-left, small

    mask_b = np.zeros((h, w), dtype=np.uint8)
    mask_b[40:60, 40:62] = 1  # bottom-right, different shape

    result = extractor.extract_batch(image, [_make_candidate(mask_a), _make_candidate(mask_b)])
    assert result.shape == (2, FEATURE_DIM)
    assert not np.array_equal(result[0], result[1]), (
        "Distinct masks should produce distinct feature vectors"
    )
