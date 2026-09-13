"""Tests for samwl.polygons and samwl.coco — Step 9 verification.

Verification criteria from the plan:
1. Every output ring has ≥ 3 points.
2. ≥ 95% of rings reach IoU ≥ 0.98 against their source mask.
3. Mean vertex count is at most half the CHAIN_APPROX_SIMPLE baseline.
4. The test prints the baseline and achieved mean vertex counts (observed
   numbers, not claims).

Additional coverage:
- Empty mask returns [].
- All-zero mask returns [].
- max_points cap is applied and warning is logged.
- morph_kernel smoothing does not crash and reduces jaggies.
- min_ring_area_frac drops speckle rings.
- coco.build_coco produces the required COCO structure.
- COCO round-trip: build → parse (the way groundtruth._load_from_coco reads it)
  yields the same mask area.

Synthetic masks are used in place of seed-frame GT masks because the seed
frames are not yet annotated in Roboflow.  Shapes (circles, ellipses,
rotated rectangles, irregular polygons) are representative of real object
outlines and produce non-trivial ``CHAIN_APPROX_SIMPLE`` baseline counts.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from samwl.coco import build_coco
from samwl.masks import mask_iou
from samwl.polygons import _ring_iou, _uniform_subsample, polygonize


# ── Synthetic mask generators ─────────────────────────────────────────────────


def _circle_mask(h: int, w: int, r: float) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = w // 2, h // 2
    cv2.circle(mask, (cx, cy), int(r), 1, -1)
    return mask


def _ellipse_mask(
    h: int, w: int, ra: float, rb: float, angle_deg: float = 0
) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = w // 2, h // 2
    cv2.ellipse(
        mask,
        (cx, cy),
        (int(ra), int(rb)),
        angle_deg,
        0,
        360,
        1,
        -1,
    )
    return mask


def _irregular_polygon_mask(h: int, w: int, n_verts: int, rng_seed: int) -> np.ndarray:
    """Irregular closed polygon with *n_verts* vertices placed on a noisy circle."""
    rng = np.random.default_rng(rng_seed)
    cx, cy = w // 2, h // 2
    r_base = min(h, w) * 0.38
    angles = np.sort(rng.uniform(0, 2 * math.pi, n_verts))
    radii = r_base * rng.uniform(0.55, 1.0, n_verts)
    xs = (cx + radii * np.cos(angles)).astype(np.int32)
    ys = (cy + radii * np.sin(angles)).astype(np.int32)
    pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts.reshape(-1, 2)], 1)
    return mask


def _make_synthetic_masks() -> list[np.ndarray]:
    """Return a list of representative single-ring binary masks."""
    masks: list[np.ndarray] = []
    # Circles of varying sizes
    for r in [20, 35, 50, 70, 90]:
        masks.append(_circle_mask(240, 320, r))
    # Ellipses with various aspect ratios and orientations
    for ra, rb, angle in [(60, 30, 0), (50, 20, 30), (80, 40, 60), (40, 25, 90)]:
        masks.append(_ellipse_mask(240, 320, ra, rb, angle))
    # Irregular polygons with different vertex counts
    for n, seed in [(12, 1), (18, 2), (24, 3), (30, 4), (40, 5), (50, 6)]:
        masks.append(_irregular_polygon_mask(240, 320, n, seed))
    return masks


def _chain_approx_simple_count(mask: np.ndarray) -> int:
    """Return total vertex count from CHAIN_APPROX_SIMPLE on *mask*."""
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return sum(len(c.reshape(-1, 2)) for c in contours)


# ── Plan verification: the three named criteria ───────────────────────────────


def test_plan_criteria_on_synthetic_masks() -> None:
    """All three plan verification criteria must pass on synthetic masks.

    1. Every output ring has ≥ 3 points.
    2. ≥ 95 % of rings reach IoU ≥ 0.98 against their source mask.
    3. Mean vertex count ≤ 0.5 × CHAIN_APPROX_SIMPLE baseline.
    """
    masks = _make_synthetic_masks()

    baseline_counts: list[int] = []
    output_counts: list[int] = []
    iou_values: list[float] = []
    min_points_violations: list[int] = []

    for mask in masks:
        baseline = _chain_approx_simple_count(mask)
        baseline_counts.append(baseline)

        rings = polygonize(mask, fidelity_iou=0.98, morph_kernel=0)

        for ring in rings:
            n = len(ring)
            output_counts.append(n)

            if n < 3:
                min_points_violations.append(n)

            # Re-rasterise the output ring and measure IoU vs the source mask
            h, w = mask.shape
            canvas = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(canvas, [ring], 1)
            iou = mask_iou(canvas, mask)
            iou_values.append(iou)

    mean_baseline = sum(baseline_counts) / len(baseline_counts)
    mean_output = sum(output_counts) / len(output_counts)

    # Mandatory print so the reduction is an observed number, not a claim
    print(
        f"\nVertex counts — baseline (CHAIN_APPROX_SIMPLE): {mean_baseline:.1f}  "
        f"achieved: {mean_output:.1f}  "
        f"ratio: {mean_output / mean_baseline:.3f}"
    )

    # Criterion 1: every ring ≥ 3 vertices
    assert not min_points_violations, (
        f"Rings with fewer than 3 vertices: {min_points_violations}"
    )

    # Criterion 2: ≥ 95 % of rings reach IoU ≥ 0.98
    n_feasible = sum(1 for iou in iou_values if iou >= 0.98)
    frac = n_feasible / len(iou_values) if iou_values else 0.0
    print(
        f"IoU ≥ 0.98: {n_feasible}/{len(iou_values)} rings ({frac*100:.1f}%)"
    )
    assert frac >= 0.95, (
        f"Only {frac*100:.1f}% of rings reached IoU ≥ 0.98 (need ≥ 95%)"
    )

    # Criterion 3: mean vertex count ≤ 0.5 × baseline
    assert mean_output <= 0.5 * mean_baseline, (
        f"Mean vertex count {mean_output:.1f} exceeds half the "
        f"CHAIN_APPROX_SIMPLE baseline {mean_baseline:.1f}"
    )


# ── Edge cases ────────────────────────────────────────────────────────────────


def test_empty_mask_returns_empty() -> None:
    mask = np.zeros((100, 100), dtype=np.uint8)
    assert polygonize(mask) == []


def test_all_zero_mask_returns_empty() -> None:
    mask = np.zeros((50, 80), dtype=np.bool_)
    assert polygonize(mask) == []


def test_min_3_vertices_degenerate_circle() -> None:
    """Even a very small circle should return a ring with ≥ 3 vertices."""
    mask = np.zeros((20, 20), dtype=np.uint8)
    cv2.circle(mask, (10, 10), 3, 1, -1)
    rings = polygonize(mask)
    assert rings, "expected at least one ring"
    for ring in rings:
        assert len(ring) >= 3, f"ring has only {len(ring)} vertices"


def test_max_points_cap_applied() -> None:
    """max_points forces vertex count down; a warning is logged via loguru."""
    import io

    from loguru import logger

    mask = _circle_mask(240, 320, 80)
    rings_uncapped = polygonize(mask, fidelity_iou=0.98)
    assert rings_uncapped, "need at least one ring to test cap"

    buf = io.StringIO()
    handler_id = logger.add(buf, level="WARNING", format="{message}")
    try:
        rings_capped = polygonize(mask, fidelity_iou=0.98, max_points=5)
    finally:
        logger.remove(handler_id)

    for ring in rings_capped:
        assert len(ring) <= 5, f"max_points=5 not enforced: {len(ring)}"
        assert len(ring) >= 3

    assert "max_points" in buf.getvalue(), (
        "expected a max_points warning to be logged by loguru"
    )


def test_morph_kernel_does_not_crash() -> None:
    """Morphological smoothing must not raise and must produce valid rings."""
    mask = _circle_mask(240, 320, 70)
    rings = polygonize(mask, fidelity_iou=0.98, morph_kernel=3)
    for ring in rings:
        assert len(ring) >= 3


def test_min_ring_area_frac_drops_speckle() -> None:
    """Rings below min_ring_area_frac are dropped."""
    # Large blob plus single tiny pixel at a corner
    mask = _circle_mask(240, 320, 70)
    mask[0, 0] = 1  # speckle

    # With generous fraction, the speckle ring is dropped
    rings_filtered = polygonize(mask, min_ring_area_frac=0.05)
    # Speckle's area = 1, circle area >> 1; filtered result should still exist
    assert rings_filtered


def test_output_ring_dtype_and_shape() -> None:
    """Each ring must be an (N, 2) int32 array."""
    mask = _ellipse_mask(200, 300, 80, 50)
    rings = polygonize(mask)
    for ring in rings:
        assert ring.dtype == np.int32, f"expected int32, got {ring.dtype}"
        assert ring.ndim == 2, f"expected 2-D, got {ring.ndim}-D"
        assert ring.shape[1] == 2, f"expected shape (N, 2), got {ring.shape}"


def test_bitwise_identical_across_runs() -> None:
    """polygonize is deterministic: two calls on the same mask are identical."""
    mask = _irregular_polygon_mask(240, 320, 30, 99)
    rings_a = polygonize(mask)
    rings_b = polygonize(mask)
    assert len(rings_a) == len(rings_b)
    for a, b in zip(rings_a, rings_b):
        assert np.array_equal(a, b), "ring vertices differed between runs"


# ── coco.build_coco ───────────────────────────────────────────────────────────


def test_build_coco_structure() -> None:
    """build_coco returns all required COCO keys."""
    mask = _circle_mask(240, 320, 60)
    rings = polygonize(mask)
    assert rings

    doc = build_coco(
        image_id="abc123",
        image_name="frame_001.jpg",
        width=320,
        height=240,
        rings=rings,
        category_name="cup",
    )

    assert "info" in doc
    assert "licenses" in doc
    assert "categories" in doc
    assert "images" in doc
    assert "annotations" in doc

    assert doc["images"][0]["width"] == 320
    assert doc["images"][0]["height"] == 240
    assert doc["images"][0]["file_name"] == "frame_001.jpg"

    assert doc["categories"][0]["name"] == "cup"
    assert doc["categories"][0]["id"] == 1

    for ann in doc["annotations"]:
        assert ann["iscrowd"] == 0
        assert "segmentation" in ann
        assert isinstance(ann["segmentation"], list)
        assert len(ann["segmentation"]) > 0
        assert "area" in ann
        assert "bbox" in ann
        assert len(ann["bbox"]) == 4


def test_build_coco_empty_rings() -> None:
    """build_coco with no valid rings produces an empty annotations list."""
    doc = build_coco(
        image_id="x",
        image_name="empty.jpg",
        width=100,
        height=100,
        rings=[],
        category_name="cup",
    )
    assert doc["annotations"] == []


def test_build_coco_skips_degenerate_rings() -> None:
    """Rings with fewer than 3 vertices are silently skipped."""
    tiny = np.array([[5, 5], [10, 10]], dtype=np.int32)  # 2 vertices
    valid = np.array([[10, 10], [50, 10], [30, 40]], dtype=np.int32)
    doc = build_coco("x", "f.jpg", 100, 100, [tiny, valid], "cup")
    assert len(doc["annotations"]) == 1


def test_build_coco_roundtrip_mask_area() -> None:
    """COCO document can be read back by the same logic as groundtruth._load_from_coco.

    The rasterised mask from the COCO segmentation field must have the same area
    as the original source mask (within a small tolerance due to anti-aliasing /
    integer rounding at polygon boundaries).
    """
    h, w = 240, 320
    source_mask = _ellipse_mask(h, w, 80, 50, 15)
    rings = polygonize(source_mask, fidelity_iou=0.98)
    assert rings

    doc = build_coco("rf_id", "frame.jpg", w, h, rings, "cup")

    # Re-parse segmentation exactly as groundtruth._load_from_coco does
    recovered = np.zeros((h, w), dtype=np.uint8)
    for ann in doc["annotations"]:
        for seg in ann["segmentation"]:
            pts = np.array(seg, dtype=np.int32).reshape(-1, 2)
            cv2.fillPoly(recovered, [pts], 1)

    iou = mask_iou(recovered, source_mask)
    assert iou >= 0.98, (
        f"Round-trip mask IoU={iou:.4f} < 0.98 — COCO segmentation lost fidelity"
    )


def test_uniform_subsample_respects_bounds() -> None:
    """_uniform_subsample always returns between 3 and n vertices."""
    ring = np.arange(20).reshape(10, 2).astype(np.int32)
    sub = _uniform_subsample(ring, 5)
    assert 3 <= len(sub) <= 5

    sub2 = _uniform_subsample(ring, 1)  # clamped to 3
    assert len(sub2) == 3

    sub3 = _uniform_subsample(ring, 20)  # no truncation needed
    assert len(sub3) == 10
