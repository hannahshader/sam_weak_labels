"""Atomic, resumable candidate mask store.

One candidate file per image: ``<candidates_dir>/<image_id>.jsonl``.

Candidate record schema (one JSON object per line)::

    {
      "image_id": str,
      "rung":     str,   # ladder rung name, e.g. "coarse"
      "rle":      {"size": [H, W], "counts": [...]},
      "bbox":     [x_min, y_min, x_max, y_max],
      "area":     int,   # pixel count
      "pred_iou": float, # SAM's predicted mask quality
      "stability_score": float
    }

Durability contract
-------------------
Candidates stream to ``<image_id>.jsonl.tmp``; each successful candidate
batch is followed by an fsync.  A ``.progress`` sidecar records one
``<image_id>\\t<byte_offset>`` line per completed image (fsync'd after the
data fsync), so a byte offset in the sidecar implies durable data up to that
offset.

On resume, ``_resume_state`` reads the sidecar, truncates the tmp file to the
last valid offset (discarding any torn tail), and returns the set of already-
completed image IDs.  When all images are done the tmp file is published via
``os.replace`` (atomic on POSIX) and the sidecar is removed.

A per-image ``generate()`` failure is logged and skipped — never fatal — so a
multi-hour run is not killed by one bad image.
"""
from __future__ import annotations

import json
import os
import signal
from pathlib import Path
from typing import Any, Callable

import numpy as np
from loguru import logger

from samwl.masks import mask_bbox, rle_encode


def _resume_state(tmp: Path, progress: Path) -> set[str]:
    """Return the set of completed image IDs and truncate any torn tail.

    Logic mirrors ``candidate_generate._resume_state`` but keyed on
    image_id strings instead of integer frame indices.
    """
    completed: set[str] = set()
    if not tmp.exists():
        progress.unlink(missing_ok=True)
        return completed
    keep = 0
    if progress.exists():
        for ln in progress.read_text().splitlines():
            try:
                image_id, off = ln.split("\t")
                completed.add(image_id)
                keep = int(off)
            except ValueError:
                break  # torn entry: everything after is untrustworthy
    size = tmp.stat().st_size
    if keep < size:
        with tmp.open("r+b") as f:
            f.truncate(keep)
    return completed


def generate_candidates(
    image_ids: list[str],
    load_image: Callable[[str], np.ndarray],
    generators: list[tuple[str, Any]],
    out_dir: Path,
    *,
    min_area: int = 5,
) -> dict[str, int]:
    """Run SAM candidate generation over a list of images and store results.

    Parameters
    ----------
    image_ids:
        Ordered list of image IDs to process.
    load_image:
        Callable that accepts an image_id and returns an H×W×3 uint8 RGB
        NumPy array.
    generators:
        List of ``(rung_name, SAM2AutomaticMaskGenerator)`` pairs, produced
        by ``samwl.sam.build_generators``.  All rungs are run for every
        image; later steps can choose how many rungs to use.
    out_dir:
        Directory where ``<image_id>.jsonl`` files are written.
    min_area:
        Candidates with fewer than this many pixels are discarded.

    Returns
    -------
    dict with keys ``total``, ``failures``, ``skipped`` (counts).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Single tmp / progress pair for the whole batch.
    tmp = out_dir / "_batch.jsonl.tmp"
    progress = out_dir / "_batch.jsonl.progress"

    done = _resume_state(tmp, progress)
    if done:
        logger.info(f"resuming, {len(done)} frames already extracted")

    # Track interruption so the finally block can distinguish clean exit.
    interrupted = False

    def _handle_sigint(sig, frame):  # noqa: ANN001
        nonlocal interrupted
        interrupted = True
        raise KeyboardInterrupt

    old_handler = signal.signal(signal.SIGINT, _handle_sigint)

    total = 0
    failures = 0
    skipped = 0

    try:
        with tmp.open("ab") as data_f, progress.open("a") as prog_f:
            for image_id in image_ids:
                final = out_dir / f"{image_id}.jsonl"
                if final.exists() or image_id in done:
                    skipped += 1
                    continue

                try:
                    rgb = load_image(image_id)
                except Exception as exc:
                    failures += 1
                    logger.error(f"{image_id}: load_image failed: {exc}")
                    continue

                image_candidates = 0
                for rung_name, gen in generators:
                    try:
                        masks = gen.generate(rgb)
                    except Exception as exc:
                        logger.error(
                            f"{image_id} rung={rung_name}: generate failed: {exc}"
                        )
                        continue

                    for c in masks:
                        seg = c["segmentation"]
                        area = int(seg.sum())
                        if area < min_area:
                            continue
                        bbox = mask_bbox(seg)
                        if not bbox:
                            continue
                        record = {
                            "image_id": image_id,
                            "rung": rung_name,
                            "rle": rle_encode(seg),
                            "bbox": bbox,
                            "area": area,
                            "pred_iou": round(float(c["predicted_iou"]), 4),
                            "stability_score": round(
                                float(c["stability_score"]), 4
                            ),
                        }
                        data_f.write((json.dumps(record) + "\n").encode())
                        image_candidates += 1

                data_f.flush()
                os.fsync(data_f.fileno())  # data durable BEFORE progress entry
                prog_f.write(f"{image_id}\t{data_f.tell()}\n")
                prog_f.flush()
                os.fsync(prog_f.fileno())

                total += image_candidates
                logger.debug(
                    f"{image_id}: {image_candidates} candidates "
                    f"({len(generators)} rung(s))"
                )
    finally:
        signal.signal(signal.SIGINT, old_handler)

    if interrupted:
        logger.warning("Interrupted mid-run; partial results are durable and resumable.")
        return {"total": total, "failures": failures, "skipped": skipped}

    # Publish: split batch tmp into per-image files, then clean up.
    _publish(out_dir, tmp, progress, done)
    return {"total": total, "failures": failures, "skipped": skipped}


def _publish(out_dir: Path, tmp: Path, progress: Path, done: set[str]) -> None:
    """Split the batch tmp file into per-image ``.jsonl`` files atomically."""
    if not tmp.exists():
        return

    per_image: dict[str, list[str]] = {}
    with tmp.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            image_id = rec.get("image_id", "")
            if image_id:
                per_image.setdefault(image_id, []).append(line)

    for image_id, lines in per_image.items():
        final = out_dir / f"{image_id}.jsonl"
        img_tmp = out_dir / f"{image_id}.jsonl.tmp"
        with img_tmp.open("w") as fh:
            for ln in lines:
                fh.write(ln + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(img_tmp, final)

    tmp.unlink(missing_ok=True)
    progress.unlink(missing_ok=True)
    logger.info(f"Published {len(per_image)} image candidate file(s) to {out_dir}")


def count(out_dir: Path, image_id: str) -> int:
    """Return the number of stored candidates for one image (0 if none)."""
    path = out_dir / f"{image_id}.jsonl"
    if not path.exists():
        return 0
    return sum(1 for line in path.open() if line.strip())


def load(out_dir: Path, image_id: str) -> list[dict]:
    """Load all candidate records for one image."""
    path = out_dir / f"{image_id}.jsonl"
    if not path.exists():
        return []
    records = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
