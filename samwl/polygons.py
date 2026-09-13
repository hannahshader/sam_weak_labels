"""Minimal-vertex polygon extraction from binary masks.

Step 9 of the SAM weak-labels plan.

Public API
----------
polygonize(mask, fidelity_iou, max_points, morph_kernel, min_ring_area_frac)
    -> list[np.ndarray]

Each returned element is an (N, 2) int32 array of (x, y) vertices with N ≥ 3.

Algorithm
---------
1. Optional morphological smoothing (open then close with a square kernel of
   side ``morph_kernel``) — removes single-pixel jaggies that inflate vertex
   counts more than epsilon tuning can recover.
2. ``RETR_EXTERNAL`` / ``CHAIN_APPROX_SIMPLE`` contour trace.
3. Drop rings whose area is below ``min_ring_area_frac`` × total mask area
   (speckle filter).  If *all* rings would be dropped, keep the largest one.
4. For each retained ring, probe 48 epsilon values geometrically from 0.1 px
   to perimeter / 2.  For each probe, apply ``approxPolyDP`` and re-rasterise;
   compute IoU against the rasterisation of the *original* (pre-simplification)
   ring.  Retain the probe with the fewest vertices among those reaching
   IoU ≥ ``fidelity_iou``.  Vertex count is not strictly monotone in epsilon so
   the search does not short-circuit on the first feasible result.
5. If no probe meets ``fidelity_iou``, fall back to the raw ``CHAIN_APPROX_SIMPLE``
   contour (maximum fidelity, maximum vertex count).
6. Floor of 3 vertices throughout.
7. If ``max_points`` is set and a ring exceeds it, apply uniform index
   subsampling to meet the cap and log a warning (fidelity may drop).
"""
from __future__ import annotations

import cv2
import numpy as np
from loguru import logger

# Number of epsilon probes for the binary search.  More probes = better chance
# of finding the true minimum vertex count despite non-monotonicity.
_N_PROBES: int = 48


def polygonize(
    mask: np.ndarray,
    fidelity_iou: float = 0.98,
    max_points: int | None = None,
    morph_kernel: int = 0,
    min_ring_area_frac: float = 0.01,
) -> list[np.ndarray]:
    """Convert a binary mask to a list of minimal-vertex polygon rings.

    Parameters
    ----------
    mask:
        2-D boolean or uint8 array.  Non-zero = foreground.
    fidelity_iou:
        Target IoU between the re-rasterised simplified ring and the original
        ring.  Default 0.98.
    max_points:
        Hard vertex cap per ring.  When set and a ring exceeds the cap, uniform
        subsampling is applied and a warning is logged; fidelity may fall below
        ``fidelity_iou``.
    morph_kernel:
        Side length in pixels of the square structuring element for an optional
        morphological open-then-close before tracing.  0 = disabled.
    min_ring_area_frac:
        Rings whose ``cv2.contourArea`` is below this fraction of the total
        non-zero pixel count of ``mask`` are treated as speckle and dropped.

    Returns
    -------
    list[np.ndarray]
        One ``(N, 2)`` int32 array per retained ring, columns (x, y).  Every
        ring has N ≥ 3.  Empty list for an all-zero mask.
    """
    mask = np.asarray(mask, dtype=np.uint8)
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {mask.shape}")

    total_area = int(mask.sum())
    if total_area == 0:
        return []

    # Step 1: optional morphological smoothing
    work = mask.copy()
    if morph_kernel > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_RECT, (morph_kernel, morph_kernel)
        )
        work = cv2.morphologyEx(work, cv2.MORPH_OPEN, k)
        work = cv2.morphologyEx(work, cv2.MORPH_CLOSE, k)

    # Step 2: contour trace on the (possibly smoothed) work mask
    contours, _ = cv2.findContours(
        work, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return []

    # Step 3: speckle filter — drop rings below threshold
    min_ring_area = min_ring_area_frac * total_area
    kept = [c for c in contours if cv2.contourArea(c) >= min_ring_area]
    if not kept:
        # All rings would be dropped; keep the largest to avoid losing the mask
        kept = [max(contours, key=cv2.contourArea)]

    rings: list[np.ndarray] = []
    for contour in kept:
        ring = _simplify_ring(contour, mask, fidelity_iou)

        # Step 7: hard vertex cap
        if max_points is not None and len(ring) > max_points:
            logger.warning(
                f"max_points={max_points} forces vertex count from "
                f"{len(ring)} down; fidelity_iou={fidelity_iou} may not hold"
            )
            ring = _uniform_subsample(ring, max_points)

        rings.append(ring)

    return rings


# ── Internal helpers ──────────────────────────────────────────────────────────


def _simplify_ring(
    contour: np.ndarray,
    source_mask: np.ndarray,
    fidelity_iou: float,
) -> np.ndarray:
    """Return the minimal-vertex simplification of *contour* meeting *fidelity_iou*.

    IoU is measured between the re-rasterised simplified ring and the
    rasterisation of the original contour (not the full source mask), so that
    multi-ring masks are handled correctly per ring.

    Falls back to the raw ``CHAIN_APPROX_SIMPLE`` contour if no probe meets
    the IoU target.
    """
    h, w = source_mask.shape
    raw = contour.reshape(-1, 2).astype(np.int32)

    if len(raw) < 3:
        # Degenerate — pad to minimum ring size
        return np.tile(raw[:1], (3, 1)).astype(np.int32)

    perimeter = cv2.arcLength(contour, closed=True)
    if perimeter < 1e-6:
        return raw

    # Rasterise the original contour to use as the per-ring reference.
    ring_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(ring_mask, [raw], 1)
    ring_area = int(ring_mask.sum())
    if ring_area == 0:
        return raw

    # Probe a geometric sequence of epsilons; retain the best feasible result
    # (fewest vertices with IoU ≥ fidelity_iou) without assuming monotonicity.
    eps_values = np.geomspace(0.1, max(perimeter / 2, 0.11), num=_N_PROBES)

    best: np.ndarray | None = None
    best_n = len(raw) + 1  # sentinel: worse than the raw contour

    for eps in eps_values:
        approx = cv2.approxPolyDP(contour, float(eps), closed=True)
        ring = approx.reshape(-1, 2).astype(np.int32)
        n = len(ring)
        if n < 3:
            continue
        iou = _ring_iou(ring, ring_mask, h, w)
        if iou >= fidelity_iou and n < best_n:
            best = ring
            best_n = n

    if best is None:
        # No probe met fidelity; fall back to the full CHAIN_APPROX_SIMPLE ring
        return raw

    return best


def _ring_iou(
    ring: np.ndarray, ref_mask: np.ndarray, h: int, w: int
) -> float:
    """IoU of a rasterised ring against *ref_mask*."""
    canvas = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(canvas, [ring], 1)
    intersection = int((canvas & ref_mask).sum())
    union = int((canvas | ref_mask).sum())
    return intersection / union if union > 0 else 0.0


def _uniform_subsample(ring: np.ndarray, n: int) -> np.ndarray:
    """Subsample *ring* to at most *n* evenly spaced vertices (minimum 3)."""
    n = max(3, n)
    k = len(ring)
    if k <= n:
        return ring
    idx = np.round(np.linspace(0, k - 1, n)).astype(int)
    return ring[idx]
