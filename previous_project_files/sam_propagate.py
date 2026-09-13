"""Shared mask helpers for the candidate pipeline.

KEEP-ONLY module: rle_decode / mask_iou / load_anchors (used by
candidate_match.py, bucket_contact_sheets.py, plot_bar_heatmap.py) and
CFG_YAML (used by candidate_generate.py). The v1 mask-propagation toolchain
(propagation main and its masks.jsonl producers) was deleted 2026-08-24 —
see WORKFLOW.md amendment 2.
"""
import json
from pathlib import Path

import cv2
import numpy as np

CFG_YAML = "configs/sam2.1/sam2.1_hiera_l.yaml"  # relative to sam2 package


def rle_decode(rle: dict) -> np.ndarray:
    h, w = rle["size"]
    counts = list(rle["counts"])
    total = h * w
    buf = np.zeros(total, dtype=np.uint8)
    pos, val = 0, 0
    for c in counts:
        buf[pos:pos + c] = val
        pos += c
        val ^= 1
    return buf.reshape((w, h)).T  # fortran order -> row-major


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def load_anchors(coco_path: Path, video_id: str):
    """Return {frame_index: (mask, bbox)} for this video from COCO export."""
    coco = json.loads(coco_path.read_text())
    img_of = {im["id"]: im for im in coco["images"]}
    anchors = {}
    for ann in coco["annotations"]:
        im = img_of[ann["image_id"]]
        name = im["file_name"]
        if not name.startswith(video_id + "_"):
            continue
        idx = int(name.split("_")[1].split(".")[0])
        h, w = im["height"], im["width"]
        mask = np.zeros((h, w), dtype=np.uint8)
        for poly in ann["segmentation"]:
            pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
            cv2.fillPoly(mask, [pts], 1)
        x, y, bw, bh = cv2.boundingRect(mask)
        anchors[idx] = (mask, (x, y, x + bw, y + bh))
    return anchors
