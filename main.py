#!/usr/bin/env python3
"""sam_weak_labels — entry point.

Subcommands
-----------
doctor  Validate config, check env, verify torch + sam2 import.
seed    Select diverse seed frames, tag them in Roboflow.   (Step 5)
fit     Generate candidates + train the mask matcher.       (Step 8)
label   Run SAM + matcher over all images, upload labels.  (Step 10)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _cmd_doctor(args: argparse.Namespace) -> int:
    """Validate config, check CUDA, verify sam2 import."""
    import pprint

    from loguru import logger
    from samwl.config import ConfigError, load

    # ── config ─────────────────────────────────────────────────────────────
    config_path = Path(args.config) if args.config else None
    try:
        cfg = load(config_path)
    except ConfigError as exc:
        logger.error(f"Config error:\n{exc}")
        return 1

    print("\n── Resolved config ─────────────────────────────────────────────")
    pprint.pprint(cfg, sort_dicts=False)

    # ── torch / CUDA ────────────────────────────────────────────────────────
    print("\n── torch / CUDA ────────────────────────────────────────────────")
    try:
        import torch

        device_str = cfg.get("sam", {}).get("device", "cuda")
        if device_str == "cuda":
            if not torch.cuda.is_available():
                logger.warning("CUDA requested but torch.cuda.is_available() is False.")
                device_name = "CPU (CUDA unavailable)"
            else:
                device_name = torch.cuda.get_device_name(0)
        else:
            device_name = f"CPU (device={device_str})"

        print(f"torch version : {torch.__version__}")
        print(f"device        : {device_name}")
    except ImportError as exc:
        logger.error(f"torch import failed: {exc}")
        return 1

    # ── sam2 ────────────────────────────────────────────────────────────────
    print("\n── sam2 ────────────────────────────────────────────────────────")
    try:
        from sam2.build_sam import build_sam2  # noqa: F401

        print("sam2 import OK")
    except ImportError as exc:
        logger.error(f"sam2 import failed: {exc}")
        return 1

    # ── weights ─────────────────────────────────────────────────────────────
    weights = Path(cfg["sam"]["weights"])
    if weights.is_file():
        size_mb = weights.stat().st_size / 1_048_576
        print(f"weights       : {weights}  ({size_mb:.0f} MB)")
    else:
        logger.warning(f"Weights file not found: {weights}")

    # ── API key presence (not value) ─────────────────────────────────────────
    import os

    api_key_set = bool(os.environ.get("ROBOFLOW_API_KEY"))
    print(
        f"\nROBOFLOW_API_KEY : {'set ✓' if api_key_set else 'NOT SET — needed for --remote, seed, fit, label'}"
    )

    # ── remote Roboflow checks (--remote) ────────────────────────────────────
    if getattr(args, "remote", False):
        print("\n── Roboflow remote ─────────────────────────────────────────────")
        from samwl.config import ConfigError, require_api_key
        from samwl.rf import (
            RFClientError,
            RFEmptyProjectError,
            RFError,
            RFSearchUnavailableError,
            RoboflowClient,
        )

        try:
            api_key = require_api_key()
        except ConfigError as exc:
            logger.error(str(exc))
            return 1

        try:
            client = RoboflowClient(cfg, api_key)
        except RFError as exc:
            logger.error(f"Roboflow connection error: {exc}")
            return 1

        # ── batch table (always works, no /search needed) ─────────────────
        try:
            batches = client.get_batches()
        except RFError as exc:
            logger.error(f"Could not fetch batches: {exc}")
            return 1

        if batches:
            col_w = max((len(b.get("name", "")) for b in batches), default=4)
            col_w = max(col_w, 4)
            header = f"  {'name':<{col_w}}  {'id':<26}  images  jobs"
            print(header)
            print("  " + "-" * (len(header) - 2))
            for b in batches:
                print(
                    f"  {b.get('name',''):<{col_w}}"
                    f"  {b.get('id',''):<26}"
                    f"  {b.get('images', '?'):>6}"
                    f"  {b.get('numJobs', '?'):>4}"
                )
        else:
            print("  (no batches)")

        # ── unassigned count (batch-derived, no /search) ──────────────────
        unassigned_n, _ = client.unassigned_summary()
        print(f"\nunassigned   : {unassigned_n}")

        # ── total image count via /search ─────────────────────────────────
        search_ok = True
        try:
            total = sum(1 for _ in client.list_images(fields=["id"]))
            print(f"total images : {total}")
        except RFEmptyProjectError:
            print("total images : 0")
            print(
                f"\n⚠  WARNING: project '{cfg['roboflow']['project_id']}' has no images. "
                "Upload images before running seed, fit, or label."
            )
        except RFSearchUnavailableError as exc:
            print(
                f"total images : (unavailable — /search returned 404; "
                f"batch-derived count is {exc.image_count})"
            )
            search_ok = False
        except RFError as exc:
            logger.error(f"Roboflow error: {exc}")
            return 1

        if not search_ok:
            print(
                "\n⚠  /search is not available on this workspace plan. "
                "Per-image enumeration (seed, label --tag) will not work "
                "until the plan is upgraded or an alternative image source is used."
            )
            print("\ndoctor: checks complete (degraded — /search unavailable) ✗")
            return 1

    # ── image fetch / cache check (--fetch N) ────────────────────────────────
    if getattr(args, "fetch", None):
        n = args.fetch
        print(f"\n── Image cache (--fetch {n}) ────────────────────────────────────")
        from pathlib import Path as _Path

        from samwl.config import REPO_ROOT
        from samwl.images import fetch_from_client, fetch_from_manifest
        from samwl.manifest import ManifestError, read as read_manifest

        cache_dir = _Path(cfg.get("data", {}).get("images_dir", "data/images"))
        if not cache_dir.is_absolute():
            cache_dir = REPO_ROOT / cache_dir

        explicit_ids = [s.strip() for s in (args.ids or "").split(",") if s.strip()]

        try:
            if explicit_ids:
                # Spot-check override: resolve via Roboflow detail endpoint.
                from samwl.config import ConfigError, require_api_key
                from samwl.rf import RoboflowClient

                try:
                    api_key = require_api_key()
                except ConfigError as exc:
                    logger.error(str(exc))
                    return 1
                client = RoboflowClient(cfg, api_key)
                results = fetch_from_client(explicit_ids[:n], client, cache_dir)
            else:
                records = read_manifest()[:n]
                if not records:
                    logger.error("Manifest is empty.")
                    return 1
                results = fetch_from_manifest(records, cache_dir)
        except ManifestError as exc:
            logger.error(str(exc))
            return 1
        except Exception as exc:
            logger.error(f"Image fetch failed: {exc}")
            return 1

        downloaded = sum(1 for r in results if not r.cached)
        cached = sum(1 for r in results if r.cached)
        for r in results:
            status = "cached" if r.cached else "downloaded"
            print(
                f"  {r.image_id}  {r.path.name}  "
                f"{r.width}x{r.height}  {r.bytes:,} B  [{status}]"
            )
        print(f"\n  {cached} cached, {downloaded} downloaded")

    # ── ground-truth coverage (--gt) ─────────────────────────────────────────
    if getattr(args, "gt", False):
        print("\n── Ground-truth coverage ────────────────────────────────────────")
        from samwl.config import REPO_ROOT as _REPO_ROOT
        from samwl.groundtruth import GroundTruthError
        from samwl.groundtruth import load as load_gt
        from samwl.manifest import ManifestError
        from samwl.manifest import read as read_manifest

        seed_ids_path = _REPO_ROOT / "data" / "seed_ids.txt"
        if not seed_ids_path.exists():
            logger.error(
                "No seed IDs found. Run `python main.py seed` first to select "
                "and tag seed frames."
            )
            return 1

        seed_ids = [
            ln.strip()
            for ln in seed_ids_path.read_text().splitlines()
            if ln.strip()
        ]
        if not seed_ids:
            logger.error(f"seed_ids.txt exists but is empty: {seed_ids_path}")
            return 1

        # Build name_by_id and dims_by_id from the manifest.
        try:
            mrecords = read_manifest()
        except ManifestError as exc:
            logger.error(f"Cannot read manifest: {exc}")
            return 1

        name_by_id = {r["image_id"]: r["name"] for r in mrecords}
        dims_by_id = {
            r["image_id"]: (r["width"], r["height"]) for r in mrecords
        }

        coco_path = Path(args.coco) if getattr(args, "coco", None) else None

        if coco_path is None:
            # API path — need a client.
            from samwl.config import ConfigError as _ConfigError
            from samwl.config import require_api_key as _require_api_key
            from samwl.rf import RoboflowClient as _RFClient

            try:
                _api_key = _require_api_key()
            except _ConfigError as exc:
                logger.error(str(exc))
                return 1
            gt_client = _RFClient(cfg, _api_key)
        else:
            gt_client = None
            if not coco_path.exists():
                logger.error(f"COCO file not found: {coco_path}")
                return 1

        try:
            gt_map = load_gt(
                seed_ids,
                client=gt_client,
                coco_path=coco_path,
                dims_by_id=dims_by_id,
                name_by_id=name_by_id,
            )
        except GroundTruthError as exc:
            logger.error(str(exc))
            return 1

        n_annotated = len(gt_map)
        n_seed = len(seed_ids)
        print(f"annotated {n_annotated} / {n_seed}")
        for image_id in seed_ids:
            if image_id in gt_map:
                gt = gt_map[image_id]
                print(
                    f"  {image_id}  polygons={gt.n_polygons}"
                    f"  area={gt.pixel_area:,} px"
                )
            else:
                print(f"  {image_id}  (no annotation)")

    print("\ndoctor: all checks passed ✓")
    return 0


def _cmd_manifest(args: argparse.Namespace) -> int:
    """Build the local image manifest (producer B)."""
    from samwl.config import ConfigError, load, require_api_key
    from samwl.manifest import ManifestError, build_from_roboflow, read
    from samwl.rf import RoboflowClient

    config_path = Path(args.config) if args.config else None
    try:
        cfg = load(config_path)
        api_key = require_api_key()
    except ConfigError as exc:
        print(f"Config error:\n{exc}", file=sys.stderr)
        return 1

    client = RoboflowClient(cfg, api_key)
    try:
        out = build_from_roboflow(client)
    except ManifestError as exc:
        print(f"manifest build failed:\n{exc}", file=sys.stderr)
        return 1

    records = read(out)
    print(f"Wrote {len(records)} records → {out}")
    sample = records[0]
    print(
        "sample:",
        {
            k: sample[k]
            for k in ("image_id", "name", "width", "height")
        },
    )
    return 0


def _cmd_seed(args: argparse.Namespace) -> int:
    """Select visually diverse seed frames and tag them in Roboflow."""
    from loguru import logger
    from tqdm import tqdm

    from samwl.config import ConfigError, load, require_api_key
    from samwl.manifest import ManifestError
    from samwl.manifest import read as read_manifest
    from samwl.seeding import select_frames

    config_path = Path(args.config) if args.config else None
    try:
        cfg = load(config_path)
    except ConfigError as exc:
        logger.error(f"Config error:\n{exc}")
        return 1

    seed_cfg = cfg.get("seed", {})
    n_frames: int = args.n if args.n is not None else seed_cfg.get("n_frames", 40)
    seed_tag: str = cfg.get("roboflow", {}).get("seed_tag", "to_manually_annotate")
    rng_seed: int = seed_cfg.get("rng_seed", 42)
    dry_run: bool = args.dry_run

    try:
        records = read_manifest()
    except ManifestError as exc:
        logger.error(str(exc))
        return 1

    logger.info(f"Manifest: {len(records)} image(s). Selecting {n_frames} frame(s).")

    try:
        selected = select_frames(records, n_frames, seed=rng_seed)
    except ValueError as exc:
        logger.error(str(exc))
        return 1

    ids = [r["image_id"] for r in selected]
    if len(ids) != len(set(ids)):
        logger.error("select_frames returned duplicate image IDs — aborting.")
        return 1

    print(
        f"\n── Seed selection: {len(selected)} frame(s) ──────────────────────────────"
    )
    for rec in selected:
        print(f"  {rec['image_id']}  {rec['name']}")

    if dry_run:
        print(f"\n[dry-run] would tag {len(selected)} image(s) with '{seed_tag}'.")
        print("No changes written.")
        return 0

    # ── tag selected images ──────────────────────────────────────────────────
    from samwl.rf import RFError, RoboflowClient

    try:
        api_key = require_api_key()
    except ConfigError as exc:
        logger.error(str(exc))
        return 1

    client = RoboflowClient(cfg, api_key)
    failed: list[str] = []
    for rec in tqdm(selected, desc="Tagging", unit="img", ncols=80):
        try:
            client.add_tags(rec["image_id"], [seed_tag])
        except RFError as exc:
            logger.error(f"Failed to tag {rec['image_id']}: {exc}")
            failed.append(rec["image_id"])

    if failed:
        logger.error(f"{len(failed)} image(s) could not be tagged.")
        return 1

    # Persist selected IDs so that doctor --gt can find the seed frames
    # without needing /search.
    from samwl.config import REPO_ROOT as _REPO_ROOT

    seed_ids_path = _REPO_ROOT / "data" / "seed_ids.txt"
    seed_ids_path.parent.mkdir(parents=True, exist_ok=True)
    seed_ids_path.write_text("\n".join(ids) + "\n")
    logger.info(f"Saved {len(ids)} seed IDs → {seed_ids_path}")

    workspace = cfg["roboflow"]["workspace"]
    project_id = cfg["roboflow"]["project_id"]
    print(f"\nTagged {len(selected)} image(s) with '{seed_tag}'.")
    print(
        "\n── Roboflow UI — next steps ─────────────────────────────────────────────"
    )
    print(
        f"  1. Open  : https://app.roboflow.com/{workspace}/{project_id}/annotate"
    )
    print(f"  2. Filter: tag = '{seed_tag}'  (should show exactly {len(selected)} images)")
    print("  3. Select all → 'Assign to job' → create a new annotation job.")
    print("  4. Annotate each image in the job, mark as Done.")
    print("  5. Run   : python main.py fit")
    print(f"\nSelected IDs: {ids}")
    return 0


def _cmd_fit(args: argparse.Namespace) -> int:
    """Train the mask matcher on annotated seed frames (Step 8)."""
    import math

    import cv2
    import numpy as np
    from loguru import logger

    from samwl import candidates as cand_store
    from samwl import matcher
    from samwl.config import REPO_ROOT, ConfigError, load, require_api_key
    from samwl.features import FeatureExtractor
    from samwl.groundtruth import GroundTruthError
    from samwl.groundtruth import load as load_gt
    from samwl.images import fetch_from_manifest
    from samwl.manifest import ManifestError
    from samwl.manifest import read as read_manifest

    config_path = Path(args.config) if args.config else None
    try:
        cfg = load(config_path)
    except ConfigError as exc:
        logger.error(f"Config error:\n{exc}")
        return 1

    # ── Seed IDs ─────────────────────────────────────────────────────────────
    seed_ids_path = REPO_ROOT / "data" / "seed_ids.txt"
    if not seed_ids_path.exists():
        logger.error("No seed IDs found. Run `python main.py seed` first.")
        return 1
    seed_ids = [
        ln.strip()
        for ln in seed_ids_path.read_text().splitlines()
        if ln.strip()
    ]
    if not seed_ids:
        logger.error("seed_ids.txt is empty.")
        return 1

    # ── Manifest ──────────────────────────────────────────────────────────────
    try:
        mrecords = read_manifest()
    except ManifestError as exc:
        logger.error(str(exc))
        return 1

    dims_by_id = {r["image_id"]: (r["width"], r["height"]) for r in mrecords}
    name_by_id = {r["image_id"]: r["name"] for r in mrecords}

    # ── Ground truth ──────────────────────────────────────────────────────────
    coco_path = Path(args.coco) if getattr(args, "coco", None) else None
    if coco_path is None:
        try:
            api_key = require_api_key()
        except ConfigError as exc:
            logger.error(str(exc))
            return 1
        from samwl.rf import RoboflowClient

        gt_client = RoboflowClient(cfg, api_key)
    else:
        gt_client = None
        if not coco_path.exists():
            logger.error(f"COCO file not found: {coco_path}")
            return 1

    try:
        gt_map = load_gt(
            seed_ids,
            client=gt_client,
            coco_path=coco_path,
            dims_by_id=dims_by_id,
            name_by_id=name_by_id,
        )
    except GroundTruthError as exc:
        logger.error(str(exc))
        return 1

    n_annotated = len(gt_map)
    logger.info(f"{n_annotated} / {len(seed_ids)} seed frames annotated.")

    if n_annotated == 0:
        logger.error(
            "No annotated seed frames found. "
            "Annotate seed frames in the Roboflow UI, then re-run `fit`."
        )
        return 1
    if n_annotated < 2:
        logger.error(
            "Leave-one-frame-out CV requires ≥ 2 annotated frames. "
            f"Found {n_annotated}. Annotate more seed frames and re-run."
        )
        return 1

    # ── Directories ───────────────────────────────────────────────────────────
    data_dir = REPO_ROOT / "data"
    images_dir = data_dir / "images"
    cands_dir = data_dir / "candidates"
    model_dir = data_dir / "model"

    matcher_cfg = cfg.get("matcher", {})
    positive_iou = float(matcher_cfg.get("positive_iou", 0.75))
    negative_iou = float(matcher_cfg.get("negative_iou", 0.25))

    annotated_ids = list(gt_map.keys())
    annotated_mrecords = [r for r in mrecords if r["image_id"] in set(annotated_ids)]

    # ── Ensure images are cached ──────────────────────────────────────────────
    logger.info(f"Ensuring {len(annotated_mrecords)} image(s) are cached…")
    try:
        fetch_from_manifest(annotated_mrecords, images_dir)
    except Exception as exc:
        logger.error(f"Image fetch failed: {exc}")
        return 1

    # ── Ensure candidates exist (generate if missing) ─────────────────────────
    missing_cands = [
        iid for iid in annotated_ids if cand_store.count(cands_dir, iid) == 0
    ]

    def _load_rgb(image_id: str) -> np.ndarray:
        for ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
            p = images_dir / f"{image_id}{ext}"
            if p.exists():
                bgr = cv2.imread(str(p))
                if bgr is not None:
                    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        raise FileNotFoundError(f"No cached image for {image_id} in {images_dir}")

    if missing_cands:
        logger.info(f"Generating candidates for {len(missing_cands)} frame(s)…")
        from samwl.sam import build_generators

        try:
            generators = build_generators(cfg)
        except Exception as exc:
            logger.error(f"Failed to build SAM generators: {exc}")
            return 1

        stats = cand_store.generate_candidates(
            missing_cands,
            _load_rgb,
            generators,
            cands_dir,
        )
        logger.info(
            f"Candidate generation complete: {stats['total']} candidates, "
            f"{stats['failures']} failures, {stats['skipped']} skipped."
        )

    # ── Load feature extractor ────────────────────────────────────────────────
    dino_model = matcher_cfg.get("embedding_model", "facebook/dinov2-base")
    device = cfg["sam"].get("device", "cuda")
    logger.info(f"Loading feature extractor ({dino_model} on {device})…")
    extractor = FeatureExtractor(model_name=dino_model, device=device)

    # ── Build frame records ───────────────────────────────────────────────────
    frame_records: list[matcher._FrameRecord] = []

    for image_id in annotated_ids:
        candidates = cand_store.load(cands_dir, image_id)
        if not candidates:
            logger.warning(f"{image_id}: no candidates found — skipping.")
            continue

        try:
            image_rgb = _load_rgb(image_id)
        except FileNotFoundError as exc:
            logger.error(str(exc))
            continue

        features = extractor.extract_batch(image_rgb, candidates)
        gt = gt_map[image_id]

        record = matcher.build_frame_record(
            image_id=image_id,
            candidates=candidates,
            gt_mask=gt.mask,
            features_all=features,
            positive_iou=positive_iou,
            negative_iou=negative_iou,
        )
        frame_records.append(record)

    if len(frame_records) < 2:
        logger.error(
            f"Need ≥ 2 frames with candidates for LOFO CV; "
            f"got {len(frame_records)}."
        )
        return 1

    # ── Pool recall per rung ──────────────────────────────────────────────────
    rung_names = [r["name"] for r in cfg["sam"]["ladder"]]
    pool_recall = matcher.compute_pool_recall(frame_records, rung_names, positive_iou)

    n_frames = len(frame_records)
    print(
        "\n── Pool recall (cumulative per rung) ───────────────────────────────────"
    )
    for rung, recall in pool_recall.items():
        n_hit = round(recall * n_frames)
        print(f"  {rung:<12}  {recall:.1%}  ({n_hit}/{n_frames} frames)")

    # ── Leave-one-frame-out CV ────────────────────────────────────────────────
    logger.info("Running leave-one-frame-out cross-validation…")
    lofo_scores = matcher.run_lofo_cv(frame_records)

    t1acc = matcher.top1_accuracy(lofo_scores, frame_records, positive_iou)
    print(f"\ntop1_accuracy (LOFO CV): {t1acc:.1%}")

    # ── Precision / coverage curve ────────────────────────────────────────────
    thresholds = np.arange(0.05, 1.0, 0.05)
    curve = matcher.precision_coverage_curve(
        lofo_scores, frame_records, positive_iou, thresholds
    )

    print("\n── Precision / coverage curve ──────────────────────────────────────────")
    print(f"  {'threshold':>9}  {'precision':>9}  {'coverage':>8}")
    for t, p, c in curve:
        p_str = f"{p:.0%}" if not math.isnan(p) else "   n/a"
        print(f"  {t:>9.2f}  {p_str:>9}  {c:>8.0%}")

    threshold = matcher.threshold_at_precision(curve, target=0.95)
    print(f"\naccept_threshold (≥95% precision): {threshold:.2f}")

    # ── Train final model ─────────────────────────────────────────────────────
    logger.info("Training final model on all annotated frames…")
    try:
        scaler, model = matcher.train_final(frame_records)
    except ValueError as exc:
        logger.error(str(exc))
        return 1

    # ── Save model + manifest ─────────────────────────────────────────────────
    n_pos_total = sum(int((r.train_labels == 1).sum()) for r in frame_records)
    n_neg_total = sum(int((r.train_labels == -1).sum()) for r in frame_records)
    metrics = {
        "n_annotated": n_frames,
        "n_candidates_pos": n_pos_total,
        "n_candidates_neg": n_neg_total,
        "pool_recall": pool_recall,
        "top1_accuracy": t1acc,
    }

    joblib_path, manifest_path = matcher.save_model(
        scaler, model, threshold, cfg, metrics, model_dir
    )

    print(
        f"\nfit complete."
        f"\n  model    → {joblib_path}"
        f"\n  manifest → {manifest_path}"
    )
    return 0


def _cmd_label(args: argparse.Namespace) -> int:
    """Run SAM + matcher over all images and upload weak labels (Step 10)."""
    from loguru import logger

    from samwl.config import REPO_ROOT, ConfigError, load, require_api_key
    from samwl.manifest import ManifestError
    from samwl.manifest import read as read_manifest
    from samwl.matcher import MatcherConfigError
    from samwl.pipeline import LabelSummary, run_label
    from samwl.rf import RFError, RoboflowClient

    config_path = Path(args.config) if args.config else None
    try:
        cfg = load(config_path)
    except ConfigError as exc:
        logger.error(f"Config error:\n{exc}")
        return 1

    dry_run: bool = args.dry_run
    limit: int | None = args.limit

    # ── Paths ─────────────────────────────────────────────────────────────────
    data_dir = REPO_ROOT / "data"
    images_dir = data_dir / "images"
    cands_dir = data_dir / "candidates"
    model_dir = data_dir / "model"
    debug_dir = data_dir / "debug"
    ledger_path = data_dir / "label_ledger.jsonl"

    # ── Run tag ───────────────────────────────────────────────────────────────
    tag_prefix: str = (
        cfg.get("roboflow", {}).get("weak_label_tag_prefix", "weak_label_v")
    )
    run_version: int = args.version if args.version is not None else 1
    run_tag = f"{tag_prefix}{run_version}"

    # ── Manifest ──────────────────────────────────────────────────────────────
    try:
        records = read_manifest()
    except ManifestError as exc:
        logger.error(str(exc))
        return 1

    if not records:
        logger.error("Manifest is empty. Run `python main.py manifest build` first.")
        return 1

    logger.info(f"Manifest: {len(records)} image(s).")

    # ── Roboflow client ───────────────────────────────────────────────────────
    if not dry_run:
        try:
            api_key = require_api_key()
        except ConfigError as exc:
            logger.error(str(exc))
            return 1
        try:
            client = RoboflowClient(cfg, api_key)
        except RFError as exc:
            logger.error(f"Roboflow connection error: {exc}")
            return 1
    else:
        # Client is required by run_label for candidate generation logging;
        # it is never called for write operations in dry-run mode.
        try:
            api_key = require_api_key()
            client = RoboflowClient(cfg, api_key)
        except (ConfigError, RFError):
            client = None  # dry-run: write path is inactive

    print(
        f"\n── label ({'dry-run, ' if dry_run else ''}run_tag={run_tag}"
        f"{f', limit={limit}' if limit else ''}) "
        "──────────────────────────────"
    )

    # ── Run pipeline ──────────────────────────────────────────────────────────
    try:
        summary: LabelSummary = run_label(
            records,
            cfg,
            client,
            dry_run=dry_run,
            limit=limit,
            run_tag=run_tag,
            ledger_path=ledger_path,
            debug_dir=debug_dir,
            images_dir=images_dir,
            cands_dir=cands_dir,
            model_dir=model_dir,
        )
    except MatcherConfigError as exc:
        logger.error(str(exc))
        return 1
    except RuntimeError as exc:
        logger.error(str(exc))
        return 1

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n── Summary ─────────────────────────────────────────────────────────")
    total_accepted = sum(summary.accepted_per_rung.values())
    for rung, n in summary.accepted_per_rung.items():
        print(f"  accepted rung={rung:<12}: {n}")
    print(f"  abstained              : {summary.abstained}")
    if dry_run:
        debug_dir_rel = debug_dir.relative_to(REPO_ROOT)
        print(f"  write requests         : 0 (dry-run)")
        print(f"  overlays + COCO files  → {debug_dir_rel}/")
    else:
        print(f"  uploaded               : {summary.uploaded}")
        print(f"  failed                 : {summary.failed}")
    print(f"  already processed      : {summary.already_processed}")
    print(f"  total accepted         : {total_accepted}")

    if dry_run:
        # Verify contract: zero write calls were made.
        write_requests = getattr(summary, "_write_requests", 0)
        assert write_requests == 0, (
            f"BUG: {write_requests} write request(s) made in dry-run mode"
        )

    if summary.failed:
        logger.warning(
            f"{summary.failed} image(s) failed — see log above. "
            "Failed images are recorded in the ledger and will be retried on re-run."
        )
        return 1

    print("\nlabel: complete ✓")
    return 0


def build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="python main.py",
        description="SAM weak labels pipeline.",
    )
    root.add_argument(
        "--config",
        metavar="PATH",
        default=None,
        help="Path to config.yaml (default: config.yaml next to main.py).",
    )
    subs = root.add_subparsers(dest="command", metavar="COMMAND")
    subs.required = True

    # doctor
    doc_p = subs.add_parser("doctor", help="Validate config, check env.")
    doc_p.set_defaults(func=_cmd_doctor)
    doc_p.add_argument(
        "--remote",
        action="store_true",
        help="Also query Roboflow (Step 2, not yet available).",
    )
    doc_p.add_argument(
        "--fetch",
        type=int,
        metavar="N",
        help="Download N images and verify the cache (Step 3).",
    )
    doc_p.add_argument(
        "--ids",
        metavar="ID1,ID2,…",
        help="Comma-separated Roboflow image IDs to override manifest for --fetch.",
    )
    doc_p.add_argument(
        "--tag",
        metavar="TAG",
        help="Also list images with this tag (Step 2, not yet available).",
    )
    doc_p.add_argument(
        "--gt",
        action="store_true",
        help="Report ground-truth annotation coverage for the seed frames (Step 6).",
    )
    doc_p.add_argument(
        "--coco",
        metavar="PATH",
        default=None,
        help=(
            "Path to a Roboflow COCO-segmentation export to use as the "
            "ground-truth source for --gt (source a).  "
            "Default: fetch urls.annotation via the API (source b)."
        ),
    )

    # manifest
    man_p = subs.add_parser(
        "manifest",
        help="Build the local image manifest (no /search).",
    )
    man_subs = man_p.add_subparsers(dest="manifest_command", metavar="COMMAND")
    man_subs.required = True
    build_p = man_subs.add_parser("build", help="Enumerate images and write data/manifest.jsonl.")
    build_p.set_defaults(func=_cmd_manifest)

    # seed (stub)
    seed_p = subs.add_parser("seed", help="Select and tag seed frames.")
    seed_p.set_defaults(func=_cmd_seed)
    seed_p.add_argument("--n", type=int, help="Override seed.n_frames from config.")
    seed_p.add_argument("--dry-run", action="store_true")

    # fit
    fit_p = subs.add_parser("fit", help="Train the mask matcher.")
    fit_p.set_defaults(func=_cmd_fit)
    fit_p.add_argument(
        "--coco",
        metavar="PATH",
        default=None,
        help=(
            "Path to a Roboflow COCO-segmentation export to use as the "
            "ground-truth source.  Default: fetch via the API."
        ),
    )

    # label
    label_p = subs.add_parser("label", help="Generate and upload weak labels.")
    label_p.set_defaults(func=_cmd_label)
    label_p.add_argument("--limit", type=int, metavar="N", help="Cap run at N images.")
    label_p.add_argument(
        "--dry-run",
        action="store_true",
        help="Write overlays/COCO locally; make no write calls to Roboflow.",
    )
    label_p.add_argument(
        "--version",
        type=int,
        metavar="N",
        default=None,
        help=(
            "Run version number appended to the tag prefix "
            "(e.g. --version 2 → weak_label_v2). Default: 1."
        ),
    )

    return root


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
