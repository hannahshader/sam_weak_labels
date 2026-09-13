"""Candidate-vs-anchor matching for the candidate pipeline (CPU).

Reads data/candidates/<vid>/candidates.jsonl (from candidate_generate.py),
filters per anchor by the area band (config `iou_bar`; the angle gate was
removed 2026-08-24 — the aligned-IoU bar subsumes it: a mask rotated past
~4-6 deg cannot hold 0.95 aligned IoU, so the window only ever rejected
noisy-angle candidates, ~+32 frames / 0.6% on the 4 videos with candidates
at the time), and scores candidates with three IoU variants against the
human anchor mask:
  aligned  anchor mask translated to the candidate's bbox center
  raw      absolute coordinates (no translation)
  scaled   candidate bbox crop resized to anchor bbox size (size-isolated)

Per-anchor output <out>/match_<vid>_<anchor>.jsonl, one line per frame:
{frame, best_aligned, best_raw, best_scaled, n_cands_area,
centroid_dist_to_anchor, best_bbox, best_angle}. --all loops the video's
anchors and prints the per-anchor + video aggregate table.

ang_delta is kept (no internal caller) because scripts/bucket_contact_sheets.py
and scripts/plot_bar_heatmap.py import it.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from pipeline_config import load_corpus  # noqa: E402
from sam_propagate import load_anchors, mask_iou, rle_decode  # noqa: E402


def ang_delta(a: float, b: float) -> float:
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def translate_mask(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Shift mask by integer (dx, dy), zero-filled, no wrap."""
    h, w = mask.shape
    out = np.zeros((h, w), dtype=mask.dtype)
    x0, x1 = max(0, dx), min(w, w + dx)
    y0, y1 = max(0, dy), min(h, h + dy)
    out[y0:y1, x0:x1] = mask[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
    return out


def scaled_iou(cand_mask: np.ndarray, anchor_mask: np.ndarray,
               anchor_bbox) -> float:
    """Candidate cropped to its bbox, resized to anchor bbox size, pasted at
    the anchor bbox origin, then IoU — isolates the size-mismatch wall."""
    ys, xs = np.nonzero(cand_mask)
    if len(xs) < 2:
        return 0.0
    crop = cand_mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    ax1, ay1, ax2, ay2 = anchor_bbox
    bw, bh = ax2 - ax1 + 1, ay2 - ay1 + 1
    if bw < 1 or bh < 1:
        return 0.0
    scaled = cv2.resize(crop.astype(np.uint8), (bw, bh),
                        interpolation=cv2.INTER_NEAREST) > 0
    placed = np.zeros_like(anchor_mask, dtype=bool)
    H, W = anchor_mask.shape
    y1, x1 = min(ay1 + bh, H), min(ax1 + bw, W)
    placed[ay1:y1, ax1:x1] = scaled[:y1 - ay1, :x1 - ax1]
    return mask_iou(placed, anchor_mask.astype(bool))


def match_anchor(vid: str, anchor_idx: int, cand_index: dict, anchor_mask,
                 anchor_bbox, area_lo: float, area_hi: float,
                 out_dir: Path) -> dict:
    anchor_area = float(anchor_mask.sum())
    anchor_cx = (anchor_bbox[0] + anchor_bbox[2]) / 2.0
    anchor_cy = (anchor_bbox[1] + anchor_bbox[3]) / 2.0
    agg = {"video": vid, "anchor": anchor_idx, "n_frames": 0,
           "n_aligned_ge_09": 0, "n_raw_ge_09": 0, "n_rescue": 0,
           "scaled_iou": [], "diverse": 0}
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"match_{vid}_{anchor_idx}.jsonl").open("w") as f:
        for frame in sorted(cand_index):
            cands = cand_index[frame]
            best = {"aligned": 0.0, "raw": 0.0, "scaled": 0.0,
                    "n_area": 0, "dist": 0.0, "bbox": None, "angle": None}
            for c in cands:
                area_ratio = c["area"] / anchor_area
                if not (area_lo <= area_ratio <= area_hi):
                    continue
                best["n_area"] += 1
                mask = rle_decode(c["rle"]).astype(bool)
                cx = (c["bbox"][0] + c["bbox"][2]) / 2.0
                cy = (c["bbox"][1] + c["bbox"][3]) / 2.0
                dx, dy = int(round(cx - anchor_cx)), int(round(cy - anchor_cy))
                aligned = mask_iou(mask, translate_mask(anchor_mask, dx, dy) > 0)
                raw = mask_iou(mask, anchor_mask.astype(bool))
                if aligned > best["aligned"]:
                    best["aligned"] = aligned
                    best["raw"] = raw
                    best["scaled"] = scaled_iou(mask, anchor_mask, anchor_bbox)
                    best["dist"] = float(np.hypot(cx - anchor_cx, cy - anchor_cy))
                    best["bbox"] = c["bbox"]
                    best["angle"] = c["angle"]
            f.write(json.dumps({
                "frame": frame, "best_aligned": round(best["aligned"], 4),
                "best_raw": round(best["raw"], 4),
                "best_scaled": round(best["scaled"], 4),
                "n_cands_area": best["n_area"],
                "centroid_dist_to_anchor": round(best["dist"], 1),
                "best_bbox": best["bbox"], "best_angle": best["angle"]}) + "\n")
            agg["n_frames"] += 1
            if best["aligned"] >= 0.9:
                agg["n_aligned_ge_09"] += 1
                if best["dist"] > 15.0:
                    agg["diverse"] += 1
            if best["raw"] >= 0.9:
                agg["n_raw_ge_09"] += 1
            if 0.7 <= best["aligned"] < 0.9:
                agg["n_rescue"] += 1
            agg["scaled_iou"].append(best["scaled"])
    agg["median_best_scaled"] = round(float(np.median(agg["scaled_iou"])), 4)
    del agg["scaled_iou"]
    n = agg["n_frames"]
    print(f"  {vid} anchor {anchor_idx}: frames={n} "
          f"aligned>=0.9={agg['n_aligned_ge_09']} ({100.0*agg['n_aligned_ge_09']/n:.1f}%) "
          f"raw>=0.9={agg['n_raw_ge_09']} rescue={agg['n_rescue']} "
          f"diverse={agg['diverse']} median_scaled={agg['median_best_scaled']}",
          flush=True)
    return agg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default=ROOT / "data" / "candidates")
    ap.add_argument("--export", default=ROOT / "data" / "roboflow-export" / "train" / "_annotations.coco.json")
    ap.add_argument("--video", required=True)
    ap.add_argument("--anchor", type=int, default=None,
                    help="single anchor frame index; omit for --all")
    ap.add_argument("--out", default=Path("/tmp/cac-pilot"))
    args = ap.parse_args()
    for p in ("candidates", "export", "out"):
        setattr(args, p, Path(getattr(args, p)))

    corpus = load_corpus()
    area_lo, area_hi = float(corpus["iou_bar"]), 1.0 / float(corpus["iou_bar"])

    cand_path = args.candidates / args.video / "candidates.jsonl"
    cand_index = {}
    for line in cand_path.read_text().splitlines():
        c = json.loads(line)
        cand_index.setdefault(c["frame"], []).append(c)

    anchors = load_anchors(args.export, args.video)
    if args.anchor is not None:
        anchor_list = [args.anchor]
    else:
        anchor_list = sorted(anchors)
    t0 = time.time()
    rows = []
    for idx in anchor_list:
        if idx not in anchors:
            print(f"anchor {idx} not in {args.video}; skip", flush=True)
            continue
        mask, bbox = anchors[idx]
        rows.append(match_anchor(args.video, idx, cand_index, mask, bbox,
                                 area_lo, area_hi, args.out))
    if args.anchor is None and rows:
        n = sum(r["n_frames"] for r in rows)
        a09 = sum(r["n_aligned_ge_09"] for r in rows)
        r09 = sum(r["n_raw_ge_09"] for r in rows)
        rescue = sum(r["n_rescue"] for r in rows)
        diverse = sum(r["diverse"] for r in rows)
        print(f"TABLE {args.video}: frames={n} "
              f"recall@0.9-aligned={100.0*a09/n:.1f}% ({a09}) "
              f"recall@0.9-raw={100.0*r09/n:.1f}% ({r09}) "
              f"rescue[0.7,0.9)={100.0*rescue/n:.1f}% ({rescue}) "
              f"diverse>15px={diverse} "
              f"median_scaled={round(float(np.median([r['median_best_scaled'] for r in rows])), 4)} "
              f"({time.time()-t0:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
