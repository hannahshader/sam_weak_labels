"""Per-candidate feature extraction for the SAM weak-labels matcher.

Feature vector layout (length FEATURE_DIM = 1556):

  Indices     Name              Description
  ──────────  ────────────────  ──────────────────────────────────────────────
  [  0: 768]  dino_fg           DINOv2 CLS — foreground crop (background zeroed)
  [768:1536]  dino_ctx          DINOv2 CLS — context crop (unmasked, same bbox)
  [1536:1556] hand-crafted      See HAND_FEATURE_NAMES (20 scalars)

The foreground and context crops are batched into a single DINOv2 forward
pass so GPU occupancy is high even when processing one image at a time.

FEATURE_DIM, DINO_DIM, HAND_DIM, and HAND_FEATURE_NAMES are module-level
constants.  The matcher manifest records them at training time so that
``label`` can refuse to run against a model built with a different layout.
"""
from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np
import torch
from loguru import logger
from transformers import AutoImageProcessor, AutoModel

from samwl.masks import rle_decode

# ── Layout constants ──────────────────────────────────────────────────────────

DINO_DIM: int = 768    # CLS-token hidden size of facebook/dinov2-base
HAND_DIM: int = 20
FEATURE_DIM: int = DINO_DIM * 2 + HAND_DIM  # 1556

# Named hand-crafted features in index order (offset from DINO_DIM * 2 = 1536).
HAND_FEATURE_NAMES: tuple[str, ...] = (
    "normalised_area",   # area / (H * W)
    "aspect_ratio",      # bbox_w / bbox_h
    "extent",            # area / bbox_area
    "solidity",          # area / convex_hull_area  (1.0 for degenerate hulls)
    "centroid_x",        # mean col of mask pixels / W
    "centroid_y",        # mean row of mask pixels / H
    "mean_H_in",         # Hue normalised to [0,1], pixels inside mask
    "std_H_in",
    "mean_S_in",         # Saturation, inside
    "std_S_in",
    "mean_V_in",         # Value, inside
    "std_V_in",
    "mean_H_out",        # Hue outside mask, within bbox
    "std_H_out",
    "mean_S_out",
    "std_S_out",
    "mean_V_out",
    "std_V_out",
    "pred_iou",          # SAM's predicted mask quality
    "stability_score",   # SAM's stability score
)
assert len(HAND_FEATURE_NAMES) == HAND_DIM


# ── Helpers ───────────────────────────────────────────────────────────────────


def _hsv_stats(hsv_norm: np.ndarray, sel: np.ndarray) -> np.ndarray:
    """Return [mean_H, std_H, mean_S, std_S, mean_V, std_V] for selected pixels.

    Parameters
    ----------
    hsv_norm:
        H'×W'×3 float32 with channels normalised to [0, 1]
        (H÷180, S÷255, V÷255).
    sel:
        H'×W' bool mask selecting which pixels to include.

    Returns
    -------
    np.ndarray of shape (6,).  All zeros when no pixels are selected.
    """
    out = np.zeros(6, dtype=np.float32)
    for c in range(3):
        vals = hsv_norm[:, :, c][sel]
        if vals.size > 0:
            out[c * 2] = float(vals.mean())
            out[c * 2 + 1] = float(vals.std())
    return out


def _solidity(mask_uint8: np.ndarray, area: int) -> float:
    """Return area / convex_hull_area.

    Uses a uint8 bbox-cropped mask.  Returns 1.0 for empty masks, single
    pixels, and any contour whose convex hull has zero area.
    """
    if area == 0:
        return 1.0
    contours, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    hull_area = 0.0
    for cnt in contours:
        if len(cnt) >= 3:
            hull = cv2.convexHull(cnt)
            hull_area += cv2.contourArea(hull)
    return area / hull_area if hull_area > 0 else 1.0


def _expanded_bbox(
    x0: int, y0: int, x1: int, y1: int, margin: float, H: int, W: int
) -> tuple[int, int, int, int]:
    """Expand bbox by a fraction of each dimension and clamp to image bounds.

    Returns (rx0, ry0, rx1, ry1) where rx1 and ry1 are exclusive.
    """
    marg_x = int(math.ceil((x1 - x0 + 1) * margin))
    marg_y = int(math.ceil((y1 - y0 + 1) * margin))
    rx0 = max(x0 - marg_x, 0)
    ry0 = max(y0 - marg_y, 0)
    rx1 = min(x1 + marg_x + 1, W)
    ry1 = min(y1 + marg_y + 1, H)
    return rx0, ry0, rx1, ry1


# ── Main class ────────────────────────────────────────────────────────────────


class FeatureExtractor:
    """Extract fixed-length feature vectors from SAM candidate masks.

    Parameters
    ----------
    model_name:
        HuggingFace model id for the vision encoder.  Must have a 768-d CLS
        token to match DINO_DIM.  Default: ``facebook/dinov2-base``.
    device:
        Torch device string (``"cuda"``, ``"cpu"``).
    crop_margin:
        Fraction of each bbox dimension added on every side before the DINOv2
        crop.  Gives the classifier visibility into surrounding context.
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        device: str = "cuda",
        crop_margin: float = 0.1,
    ) -> None:
        self._device = torch.device(device)
        self._crop_margin = crop_margin
        self.model_name = model_name

        logger.info(f"Loading DINOv2 processor+model: {model_name}")
        self._processor = AutoImageProcessor.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name)
        self._model.eval()
        self._model.to(self._device)
        logger.info(f"DINOv2 ready on {self._device}")

    def extract_batch(
        self,
        image_rgb: np.ndarray,
        candidates: list[dict[str, Any]],
    ) -> np.ndarray:
        """Extract feature vectors for all candidates from one image.

        Parameters
        ----------
        image_rgb:
            H×W×3 uint8 NumPy array (RGB).
        candidates:
            List of candidate dicts as stored by ``samwl.candidates``.  Each
            must have: ``rle``, ``bbox``, ``area``, ``pred_iou``,
            ``stability_score``.

        Returns
        -------
        np.ndarray
            Shape ``(N, FEATURE_DIM)`` float32, in the same order as
            ``candidates``.  Returns ``(0, FEATURE_DIM)`` when the list is
            empty.
        """
        if not candidates:
            return np.empty((0, FEATURE_DIM), dtype=np.float32)

        H, W = image_rgb.shape[:2]

        # Normalised HSV: H÷180 → [0,1],  S÷255 → [0,1],  V÷255 → [0,1].
        hsv_norm = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv_norm[:, :, 0] /= 180.0
        hsv_norm[:, :, 1] /= 255.0
        hsv_norm[:, :, 2] /= 255.0

        hand_block = np.zeros((len(candidates), HAND_DIM), dtype=np.float32)
        fg_crops: list[np.ndarray] = []
        ctx_crops: list[np.ndarray] = []

        for i, cand in enumerate(candidates):
            mask = rle_decode(cand["rle"]).astype(bool)  # H×W bool
            x0, y0, x1, y1 = cand["bbox"]
            area = int(cand["area"])

            # ── geometry ──────────────────────────────────────────────────
            bw = max(x1 - x0 + 1, 1)
            bh = max(y1 - y0 + 1, 1)
            bbox_area = bw * bh

            ys, xs = np.nonzero(mask)
            if xs.size > 0:
                cx = float(xs.mean())
                cy = float(ys.mean())
            else:
                cx = (x0 + x1) / 2.0
                cy = (y0 + y1) / 2.0

            # Bbox crop (uint8) for contour-based solidity.
            # Offset from origin doesn't affect contour areas.
            bbox_crop_uint8 = mask[y0 : y1 + 1, x0 : x1 + 1].astype(np.uint8)
            sol = _solidity(bbox_crop_uint8, area)

            # ── colour ────────────────────────────────────────────────────
            # Clamp to image bounds (bbox coords are always in-range but guard
            # against floating-point-rounded-bbox edge cases).
            cy0 = max(y0, 0)
            cy1 = min(y1 + 1, H)
            cx0 = max(x0, 0)
            cx1 = min(x1 + 1, W)
            mask_crop = mask[cy0:cy1, cx0:cx1]
            hsv_crop = hsv_norm[cy0:cy1, cx0:cx1]

            in_stats = _hsv_stats(hsv_crop, mask_crop)
            out_stats = _hsv_stats(hsv_crop, ~mask_crop)

            # ── hand-crafted vector ───────────────────────────────────────
            hand_block[i, :] = [
                area / (H * W),         # normalised_area
                bw / bh,                # aspect_ratio
                area / bbox_area,       # extent
                sol,                    # solidity
                cx / W,                 # centroid_x
                cy / H,                 # centroid_y
                *in_stats,              # mean/std H S V inside mask
                *out_stats,             # mean/std H S V outside mask (within bbox)
                float(cand["pred_iou"]),
                float(cand["stability_score"]),
            ]

            # ── DINOv2 crops ──────────────────────────────────────────────
            rx0, ry0, rx1, ry1 = _expanded_bbox(
                x0, y0, x1, y1, self._crop_margin, H, W
            )
            ctx = image_rgb[ry0:ry1, rx0:rx1]   # H'×W'×3 uint8 (unmasked)
            fg_mask = mask[ry0:ry1, rx0:rx1]
            fg = ctx.copy()
            fg[~fg_mask] = 0                     # zero out background

            fg_crops.append(fg)
            ctx_crops.append(ctx)

        # ── DINOv2 forward pass (fg crops first, then ctx crops) ──────────
        all_crops: list[np.ndarray] = fg_crops + ctx_crops  # 2N items
        inputs = self._processor(images=all_crops, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        # CLS token is at sequence position 0.
        cls = outputs.last_hidden_state[:, 0, :].float().cpu().numpy()
        n = len(candidates)
        fg_emb = cls[:n]   # (N, DINO_DIM)
        ctx_emb = cls[n:]  # (N, DINO_DIM)

        result = np.concatenate([fg_emb, ctx_emb, hand_block], axis=1)
        assert result.shape == (n, FEATURE_DIM), (
            f"expected ({n}, {FEATURE_DIM}), got {result.shape}"
        )
        return result.astype(np.float32)
