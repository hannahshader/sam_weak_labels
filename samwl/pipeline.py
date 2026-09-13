"""Label pipeline: run SAM + matcher over target images, upload weak labels.

Step 10 of the SAM weak-labels plan.

Per-image workflow
------------------
1. Skip if the image ID is already in the progress ledger.
2. Ensure the image is cached locally (download if needed).
3. Ensure SAM candidates exist for the image; generate all rungs if absent.
4. Extract DINOv2 + hand-crafted features for all candidates.
5. Score candidates with the trained logistic-regression matcher.
6. Escalate through ladder rungs: at rung k, consider candidates from
   rungs 0..k; accept the argmax if its probability >= accept_threshold.
7. Abstain when the ladder is exhausted without reaching the threshold.
8. Polygonize the accepted mask, build a single-image COCO document.
9. Write a PNG overlay to ``debug_dir`` (always).
10. Write the COCO JSON to ``debug_dir`` (always).
11. Unless dry_run: call ``client.save_weak_label(image_id, coco_path, run_tag)``.
12. Append a ledger entry for the image.

Public API
----------
run_label(records, cfg, client, *, dry_run, limit, run_tag,
          ledger_path, debug_dir, images_dir, cands_dir, model_dir)
    -> LabelSummary
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from loguru import logger

from samwl import candidates as cand_store
from samwl.coco import build_coco
from samwl.features import FeatureExtractor
from samwl.masks import rle_decode
from samwl.polygons import polygonize


# ── Summary type ──────────────────────────────────────────────────────────────


@dataclass
class LabelSummary:
    """Aggregate counts returned by ``run_label``."""

    #: How many images were accepted at each rung (order matches the ladder).
    accepted_per_rung: dict[str, int] = field(default_factory=dict)
    abstained: int = 0
    uploaded: int = 0
    failed: int = 0
    already_processed: int = 0
    dry_run: bool = False


# ── Ledger helpers ────────────────────────────────────────────────────────────


def _load_ledger(path: Path) -> set[str]:
    """Return the set of image IDs already recorded in the ledger."""
    if not path.exists():
        return set()
    done: set[str] = set()
    for line in path.open():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            done.add(rec["image_id"])
        except (json.JSONDecodeError, KeyError):
            continue
    return done


def _write_ledger_entry(path: Path, image_id: str, status: str, rung: str | None) -> None:
    """Append one JSON line to the ledger (fsync'd for durability)."""
    entry = {
        "image_id": image_id,
        "status": status,
        "rung": rung,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(entry) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


# ── Overlay helper ────────────────────────────────────────────────────────────


def _write_overlay(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    out_path: Path,
    alpha: float = 0.45,
    color: tuple[int, int, int] = (0, 230, 90),
) -> None:
    """Write a debug PNG with the accepted mask drawn as a semi-transparent overlay.

    Parameters
    ----------
    image_rgb:
        H×W×3 uint8 RGB image.
    mask:
        H×W bool/uint8 array; non-zero = foreground.
    out_path:
        Destination path (PNG).
    alpha:
        Opacity of the overlay (0 = invisible, 1 = opaque).
    color:
        BGR colour of the overlay (default: green).
    """
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    overlay = bgr.copy()
    m = mask.astype(bool)
    overlay[m] = (
        (1 - alpha) * bgr[m].astype(np.float32)
        + alpha * np.array(color, dtype=np.float32)
    ).clip(0, 255).astype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)


# ── Core label pipeline ───────────────────────────────────────────────────────


def run_label(
    records: list[dict],
    cfg: dict[str, Any],
    client: Any,
    *,
    dry_run: bool,
    limit: int | None,
    run_tag: str,
    ledger_path: Path,
    debug_dir: Path,
    images_dir: Path,
    cands_dir: Path,
    model_dir: Path,
) -> LabelSummary:
    """Run the label pipeline over ``records``.

    Parameters
    ----------
    records:
        Manifest records (all target images); each has at least ``image_id``,
        ``name``, ``width``, ``height``, ``source``, and either ``url`` or
        ``path``.
    cfg:
        Pipeline config from ``samwl.config.load()``.
    client:
        A ``samwl.rf.RoboflowClient`` instance.  Not used when ``dry_run``
        is True but still required (for candidate generation logging).
    dry_run:
        When True, write overlays and COCO files but make no write calls.
    limit:
        Cap the run at this many images (None = no cap).
    run_tag:
        Roboflow tag added to every uploaded image, e.g. ``"weak_label_v1"``.
    ledger_path:
        Path to the JSONL progress ledger.  Each line is a JSON object with
        ``image_id``, ``status``, ``rung``, and ``ts``.
    debug_dir:
        Directory to write overlay PNGs and COCO JSON files.
    images_dir:
        Image cache directory (``data/images``).
    cands_dir:
        Candidate store directory (``data/candidates``).
    model_dir:
        Model directory containing ``matcher.joblib`` and ``manifest.json``.

    Returns
    -------
    LabelSummary
    """
    from samwl import matcher
    from samwl.images import fetch_from_manifest
    from samwl.sam import build_generators

    # ── Load matcher ──────────────────────────────────────────────────────────
    try:
        scaler, model, threshold, manifest = matcher.load_model(model_dir, cfg)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Matcher model not found: {exc}\n"
            "Run `python main.py fit` first."
        ) from exc

    dino_model: str = manifest.get("dino_model", "facebook/dinov2-base")
    device: str = cfg["sam"].get("device", "cuda")
    extractor = FeatureExtractor(model_name=dino_model, device=device)

    ladder: list[dict] = cfg["sam"]["ladder"]
    rung_names: list[str] = [r["name"] for r in ladder]

    summary = LabelSummary(
        accepted_per_rung={r: 0 for r in rung_names},
        dry_run=dry_run,
    )

    # ── Resume ledger ─────────────────────────────────────────────────────────
    done = _load_ledger(ledger_path)
    if done:
        logger.info(f"{len(done)} image(s) already in ledger — resuming.")

    # ── Filter records ────────────────────────────────────────────────────────
    target_records = records
    already = [r for r in target_records if r["image_id"] in done]
    summary.already_processed = len(already)
    pending = [r for r in target_records if r["image_id"] not in done]

    if limit is not None:
        pending = pending[:limit]

    if not pending:
        logger.info("No images to process.")
        return summary

    logger.info(
        f"Processing {len(pending)} image(s) "
        f"({'dry-run, ' if dry_run else ''}limit={limit})."
    )

    # ── Ensure images are cached ──────────────────────────────────────────────
    logger.info(f"Ensuring {len(pending)} image(s) are cached…")
    try:
        fetch_results = fetch_from_manifest(pending, images_dir)
    except Exception as exc:
        raise RuntimeError(f"Image fetch failed: {exc}") from exc

    fetch_by_id = {r.image_id: r for r in fetch_results}

    def _load_rgb(image_id: str) -> np.ndarray:
        for ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
            p = images_dir / f"{image_id}{ext}"
            if p.exists():
                bgr = cv2.imread(str(p))
                if bgr is not None:
                    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        raise FileNotFoundError(f"No cached image for {image_id} in {images_dir}")

    # ── Ensure candidates exist (generate any that are missing) ───────────────
    pending_ids = [r["image_id"] for r in pending]
    missing_cands = [iid for iid in pending_ids if cand_store.count(cands_dir, iid) == 0]

    if missing_cands:
        logger.info(f"Generating candidates for {len(missing_cands)} image(s)…")
        try:
            generators = build_generators(cfg)
        except Exception as exc:
            raise RuntimeError(f"Failed to build SAM generators: {exc}") from exc

        stats = cand_store.generate_candidates(
            missing_cands,
            _load_rgb,
            generators,
            cands_dir,
        )
        logger.info(
            f"Candidate generation: {stats['total']} candidates, "
            f"{stats['failures']} failure(s), {stats['skipped']} skipped."
        )

    # ── Per-image scoring and upload ──────────────────────────────────────────
    polygon_cfg = cfg.get("polygon", {})
    fidelity_iou: float = float(polygon_cfg.get("fidelity_iou", 0.98))
    max_points_raw = polygon_cfg.get("max_points")
    max_points: int | None = int(max_points_raw) if max_points_raw is not None else None
    morph_kernel: int = int(polygon_cfg.get("morph_kernel", 3))
    min_ring_area_frac: float = float(polygon_cfg.get("min_ring_area_frac", 0.01))

    category_name: str = cfg["target"]["class_name"]
    category_id: int = int(cfg["target"].get("category_id", 1))

    debug_dir.mkdir(parents=True, exist_ok=True)

    write_requests = 0  # used to verify zero writes in dry-run

    for rec in pending:
        image_id: str = rec["image_id"]
        image_name: str = rec["name"]
        width: int = int(rec["width"])
        height: int = int(rec["height"])

        try:
            # ── Load image ────────────────────────────────────────────────────
            try:
                image_rgb = _load_rgb(image_id)
            except FileNotFoundError as exc:
                raise RuntimeError(str(exc)) from exc

            # ── Load candidates ───────────────────────────────────────────────
            candidates = cand_store.load(cands_dir, image_id)
            if not candidates:
                raise RuntimeError(f"{image_id}: no candidates found after generation")

            # ── Extract features (all candidates at once) ─────────────────────
            features = extractor.extract_batch(image_rgb, candidates)

            # ── Score all candidates ──────────────────────────────────────────
            X_scaled = scaler.transform(features)
            proba = model.predict_proba(X_scaled)
            pos_idx = list(model.classes_).index(1)
            scores = proba[:, pos_idx].astype(np.float32)

            # ── Rung-based escalation ─────────────────────────────────────────
            # Build a mask for each cumulative rung prefix.
            cand_rungs = [c["rung"] for c in candidates]
            accepted_idx: int | None = None
            accepted_rung: str | None = None

            for k, rung in enumerate(rung_names):
                rung_prefix = set(rung_names[: k + 1])
                prefix_mask = np.array(
                    [r in rung_prefix for r in cand_rungs], dtype=bool
                )
                if not prefix_mask.any():
                    # No candidates from this rung prefix — skip to next.
                    continue
                prefix_scores = np.where(prefix_mask, scores, -np.inf)
                best_idx = int(np.argmax(prefix_scores))
                best_score = float(scores[best_idx])
                if best_score >= threshold:
                    accepted_idx = best_idx
                    accepted_rung = rung
                    break

            if accepted_idx is None:
                # Ladder exhausted — abstain.
                logger.info(f"{image_id}: abstained (best score below threshold)")
                summary.abstained += 1
                _write_ledger_entry(ledger_path, image_id, "abstained", None)
                continue

            summary.accepted_per_rung[accepted_rung] += 1

            # ── Polygonize the accepted mask ──────────────────────────────────
            accepted_cand = candidates[accepted_idx]
            mask = rle_decode(accepted_cand["rle"])

            rings = polygonize(
                mask,
                fidelity_iou=fidelity_iou,
                max_points=max_points,
                morph_kernel=morph_kernel,
                min_ring_area_frac=min_ring_area_frac,
            )

            # ── Build COCO document ───────────────────────────────────────────
            coco_doc = build_coco(
                image_id=image_id,
                image_name=image_name,
                width=width,
                height=height,
                rings=rings,
                category_name=category_name,
                category_id=category_id,
            )

            # ── Write debug artifacts ─────────────────────────────────────────
            overlay_path = debug_dir / f"{image_id}_overlay.png"
            _write_overlay(image_rgb, mask, overlay_path)

            coco_path = debug_dir / f"{image_id}.json"
            coco_path.write_text(json.dumps(coco_doc, indent=2))

            logger.debug(
                f"{image_id}: accepted rung={accepted_rung} "
                f"score={scores[accepted_idx]:.3f} "
                f"rings={len(rings)}"
            )

            # ── Upload (non-dry-run) ──────────────────────────────────────────
            if not dry_run:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    suffix=".json",
                    delete=False,
                    dir=debug_dir,
                ) as tmp_fh:
                    tmp_path = Path(tmp_fh.name)
                    json.dump(coco_doc, tmp_fh)

                try:
                    client.save_weak_label(image_id, tmp_path, run_tag)
                    write_requests += 1
                    summary.uploaded += 1
                finally:
                    tmp_path.unlink(missing_ok=True)

            _write_ledger_entry(ledger_path, image_id, "accepted", accepted_rung)

        except Exception as exc:
            logger.error(f"{image_id}: failed — {exc}")
            summary.failed += 1
            _write_ledger_entry(ledger_path, image_id, "failed", None)

    # Expose write_requests count on the summary for dry-run assertions.
    summary._write_requests = write_requests  # type: ignore[attr-defined]
    return summary
