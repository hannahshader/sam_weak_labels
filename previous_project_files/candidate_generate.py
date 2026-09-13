"""Grid autogen candidate pass for the candidate-pilot.

Runs SAM2AutomaticMaskGenerator (in-venv sam-2==1.0) over videos and
writes every candidate mask to data/candidates/<vid>/candidates.jsonl:
  {"frame": int, "rle": {...}, "bbox": [x1,y1,x2,y2], "angle": float,
   "area": int, "pred_iou": float}

Atomic + resumable: candidates stream to candidates.jsonl.tmp with a
candidates.jsonl.progress sidecar (one 'frame\\tbyte_offset' line per
completed frame, fsynced after that frame's data). The final file is
published by atomic rename, so it is never a partial file; an interrupted
run resumes on the next invocation (completed frames skipped, torn tails
truncated). A video whose candidates.jsonl already exists is skipped
entirely (never re-extracted); delete the file to force a fresh run.

Generator seam: _build_generator() — swap here if the SAM3 research verdict
lands on USE-SAM3 (same generate(rgb) -> [dicts] contract).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

PILOT_VIDEOS = ["0278afd4", "2b08c1b1", "8ddce56d"]


def rle_fast(mask: np.ndarray) -> dict:
    """Vectorized COCO RLE — byte-identical to the pure-Python encoder
    removed from sam_propagate with the v1 toolchain, but numpy runs, not
    a Python per-pixel loop (50 masks/frame at 5M px makes
    the pure-Python version the pipeline bottleneck: ~10 s/frame)."""
    flat = np.asfortranarray(mask.astype(np.uint8)).ravel(order="F")
    n = flat.size
    idx = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    starts = np.concatenate(([0], idx))
    ends = np.concatenate((idx, [n]))
    counts = (ends - starts).tolist()
    if flat[0] == 1:
        counts = [0] + counts
    return {"size": [int(mask.shape[0]), int(mask.shape[1])], "counts": counts}


def _build_generator(model):
    """Seam: SAM2AutomaticMaskGenerator now; SAM3 equivalent if verdict says so."""
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    return SAM2AutomaticMaskGenerator(model)


def mask_angle(mask: np.ndarray) -> float:
    """Min-area-rect angle of the largest external contour, mod 180."""
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key=cv2.contourArea)
    return float(cv2.minAreaRect(cnt)[2] % 180.0)


def mask_bbox(mask: np.ndarray):
    """xyxy from nonzero extents — format-independent (SAM dict bbox is not trusted)."""
    ys, xs = np.nonzero(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def _resume_state(tmp: Path, progress: Path) -> set:
    """Resume bookkeeping for one video's candidate extraction.

    tmp holds this run's candidates as they stream; progress has one line
    per completed frame, 'frame\\tbyte_offset', where byte_offset is the tmp
    size AFTER that frame's candidates were fsynced (so a durable entry
    implies durable data). Returns the completed frame set and truncates
    tmp to the last valid entry's offset — anything after it is a torn
    frame (crash mid-write) or a torn entry and is discarded. A tmp
    without progress means no frame ever completed: wipe it. A stale
    progress without tmp (crash between publish and cleanup) is dropped.
    """
    completed = set()
    if not tmp.exists():
        progress.unlink(missing_ok=True)
        return completed
    keep = 0
    if progress.exists():
        for ln in progress.read_text().splitlines():
            try:
                fr, off = ln.split("\t")
                completed.add(int(fr))
                keep = int(off)
            except ValueError:
                break  # torn entry: later entries are untrustworthy
    size = tmp.stat().st_size
    if keep < size:  # never extend (power loss can leave tmp shorter)
        with tmp.open("r+b") as f:
            f.truncate(keep)
    return completed


def run_video(vid: str, gen, frames_dir: Path, out_dir: Path,
              max_frames: int, frame_range: tuple) -> dict:
    frames = sorted(frames_dir.glob(f"{vid}_*.jpg"))
    if frame_range:
        lo, hi = frame_range
        frames = [f for f in frames if lo <= int(f.stem.split("_")[1]) <= hi]
    n = len(frames)
    if max_frames:
        frames = frames[:max_frames]
    total, failures, t0 = 0, 0, time.time()
    vid_dir = out_dir / vid
    vid_dir.mkdir(parents=True, exist_ok=True)
    tmp = vid_dir / "candidates.jsonl.tmp"
    progress = vid_dir / "candidates.jsonl.progress"
    final = vid_dir / "candidates.jsonl"
    if final.exists():
        # completed run is authoritative; drop any stale resume state
        tmp.unlink(missing_ok=True)
        progress.unlink(missing_ok=True)
        print(f"  {vid}: skipping, candidates.jsonl already exists", flush=True)
        return {"video": vid, "frames": 0, "candidates": 0,
                "failures": 0, "seconds": 0.0, "skipped": True}
    done = _resume_state(tmp, progress)
    if done:
        print(f"  {vid}: resuming, {len(done)} frames already extracted",
              flush=True)
    with tmp.open("ab") as f, progress.open("a") as p:
        for k, fp in enumerate(frames, start=1):
            idx = int(fp.stem.split("_")[1])
            if idx in done:
                continue
            img = cv2.imread(str(fp))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            try:
                cands = gen.generate(img)
            except Exception as e:  # one bad frame must not kill a multi-hour run
                failures += 1
                print(f"  {vid} frame {idx}: generate failed: {e}", flush=True)
                continue
            for c in cands:
                mask = c["segmentation"]
                if mask.sum() < 5:
                    continue
                f.write((json.dumps({
                    "frame": idx,
                    "rle": rle_fast(mask),
                    "bbox": mask_bbox(mask),
                    "angle": round(mask_angle(mask), 2),
                    "area": int(mask.sum()),
                    "pred_iou": round(float(c["predicted_iou"]), 3),
                }) + "\n").encode())
                total += 1
            f.flush()
            os.fsync(f.fileno())  # frame durable BEFORE its progress entry
            p.write(f"{idx}\t{f.tell()}\n")
            p.flush()
            os.fsync(p.fileno())
            if k % 25 == 0:
                print(f"  {vid} frame {idx}/{n} cand_so_far={total}", flush=True)
    # publish: candidates.jsonl appears complete-or-not-at-all
    os.replace(tmp, vid_dir / "candidates.jsonl")
    progress.unlink(missing_ok=True)
    secs = round(time.time() - t0, 1)
    resumed = f" (+{len(done)} resumed)" if done else ""
    print(f"{vid}: {n} frames, {total} candidates{resumed}, "
          f"{failures} failures, {secs}s", flush=True)
    return {"video": vid, "frames": n, "candidates": total,
            "failures": failures, "seconds": secs}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", default=PILOT_VIDEOS)
    ap.add_argument("--weights", default=ROOT / "models" / "sam2.1_hiera_large.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default=ROOT / "data" / "candidates")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="limit frames per video (0 = all)")
    ap.add_argument("--frame-range", default=None,
                    help="inclusive 'A-B' frame filter (e.g. 148-157); for the "
                         "MEASURE-BOTH sample only")
    args = ap.parse_args()

    from sam2.build_sam import build_sam2
    from sam_propagate import CFG_YAML

    fr = tuple(map(int, args.frame_range.split("-"))) if args.frame_range else None
    print(f"generator=SAM2AutomaticMaskGenerator(defaults) "
          f"weights={args.weights} device={args.device} "
          f"videos={args.videos} frame_range={fr}", flush=True)
    model = build_sam2(CFG_YAML, str(args.weights), device=args.device)
    gen = _build_generator(model)

    t0 = time.time()
    for vid in args.videos:
        run_video(vid, gen, ROOT / "data" / "videos" / vid / "frames",
                  args.out_dir, args.max_frames, fr)
    print(f"done in {time.time() - t0:.1f}s total", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
