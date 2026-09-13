"""Certify the training corpus from candidate-vs-anchor match files (v2).

Reads every match_<vid>_<anchor>.jsonl under --matches (produced by
candidate_match.py), attributes each frame to its best anchor — argmax of
best_aligned, tie -> smallest anchor index — and keeps the frame iff
best_aligned >= iou_bar (config.json["corpus"]["iou_bar"]). No caps, no
angle certification bar, no exclusions. Writes the corpus manifest and the
attributed-frame angle distribution (PNG + CSV + printed stats).

Deterministic: no rng, sorted iteration; re-running yields a byte-identical
corpus.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from pipeline_config import load_corpus  # noqa: E402

BIN_W = 5.0  # degrees


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", default=ROOT / "data" / "candidates" / "matches")
    ap.add_argument("--out", default=ROOT / "data" / "training" / "corpus.jsonl")
    args = ap.parse_args()
    corpus = load_corpus()
    iou_bar = float(corpus["iou_bar"])

    matches = defaultdict(dict)  # vid -> anchor idx -> {frame: row}
    for fp in sorted(Path(args.matches).glob("match_*.jsonl")):
        _, vid, anchor = fp.stem.split("_")
        rows = {int(r["frame"]): r
                for r in (json.loads(ln) for ln in fp.read_text().splitlines())}
        matches[vid][int(anchor)] = rows
    if not matches:
        print(f"no match files under {args.matches}", file=sys.stderr)
        return 1

    frames_by_vid = defaultdict(set)
    best = {}  # (vid, frame) -> [best_aligned, anchor, angle, bbox]
    for vid in sorted(matches):
        for aidx in sorted(matches[vid]):  # ascending -> tie keeps smallest idx
            for frame in sorted(matches[vid][aidx]):
                frames_by_vid[vid].add(frame)
                row = matches[vid][aidx][frame]
                key = (vid, frame)
                if key not in best or row["best_aligned"] > best[key][0]:
                    best[key] = (row["best_aligned"], aidx,
                                 row["best_angle"], row["best_bbox"])

    attributed = [(vid, frame, aligned, aidx, angle, bbox)
                  for (vid, frame), (aligned, aidx, angle, bbox)
                  in sorted(best.items()) if aligned >= iou_bar]

    # invariants (exit 1 on violation)
    assert len(best) == sum(len(f) for f in frames_by_vid.values()), "frame union mismatch"
    assert len({(v, f) for v, f, *_ in attributed}) == len(attributed), "duplicate (video, frame)"
    assert all(a >= iou_bar for _, _, a, *_ in attributed), "bar violation"
    assert all(angle is not None for _, _, _, _, angle, _ in attributed), "None angle attributed"
    per_video_attr = defaultdict(int)
    for vid, *_ in attributed:
        per_video_attr[vid] += 1
    for vid in frames_by_vid:
        assert per_video_attr[vid] <= len(frames_by_vid[vid]), "attributed > frames"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for vid, frame, aligned, aidx, angle, bbox in attributed:
            f.write(json.dumps({
                "video": vid, "frame": frame, "anchor": aidx,
                "aligned_iou": round(aligned, 4), "angle": round(angle, 2),
                "bbox": bbox,
            }) + "\n")

    # report: per-video attributed/total/unattributed + anchor counts
    by_vid = defaultdict(list)
    for r in attributed:
        by_vid[r[0]].append(r)
    for vid in sorted(frames_by_vid):
        rows = by_vid[vid]
        total = len(frames_by_vid[vid])
        print(f"{vid}: {len(rows)}/{total} attributed, {total - len(rows)} unattributed")
        anchor_counts = defaultdict(int)
        for r in rows:
            anchor_counts[r[3]] += 1
        for aidx in sorted(anchor_counts):
            print(f"  anchor {aidx}: {anchor_counts[aidx]}")
    print(f"TOTAL: {len(attributed)} frames | {len(frames_by_vid)} videos")

    # distribution of attributed angles (never recomputed from rle)
    angles = np.array([r[4] for r in attributed], dtype=float)
    bins = np.arange(0.0, 180.0 + BIN_W, BIN_W)
    counts, _ = np.histogram(angles, bins=bins)
    total = len(angles)
    cum = np.cumsum(counts)
    with (out_path.parent / "angle-distribution.csv").open("w") as f:
        f.write("bin_lo,bin_hi,count,cum_pct\n")
        for lo, hi, c, cu in zip(bins[:-1], bins[1:], counts, cum):
            pct = 100.0 * cu / total if total else 0.0
            f.write(f"{lo:.1f},{hi:.1f},{c},{pct:.2f}\n")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(angles, bins=bins)
    ax.set_xlabel("angle (deg)")
    ax.set_ylabel("attributed frames")
    ax.set_title(f"corpus v2: {total} attributed frames | IoU >= {iou_bar:g} | no angle gate")
    ax.set_xlim(0, 180)
    fig.tight_layout()
    fig.savefig(out_path.parent / "angle-distribution.png", dpi=150)
    plt.close(fig)

    if total:
        print(f"angle stats (attributed): n={total} "
              f"mean={angles.mean():.2f} median={np.median(angles):.2f} "
              f"p10={np.percentile(angles, 10):.2f} p90={np.percentile(angles, 90):.2f} "
              f"min={angles.min():.2f} max={angles.max():.2f}")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
