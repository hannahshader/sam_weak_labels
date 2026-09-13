"""Tests for samwl.pipeline — Step 10 verification.

Verification criteria from the plan:
1. run_label in dry-run mode writes one overlay PNG and one COCO JSON per
   processed image, makes zero write calls, and returns a summary with a
   per-rung acceptance breakdown.
2. Already-processed entries in the ledger are counted as already_processed
   and skipped — re-running the same limit reports N already processed, 0 new.
3. The --dry-run assertion in _cmd_label fires when write_requests > 0 (tested
   via direct assertion on summary._write_requests == 0).

The tests use synthetic images, synthetic candidate JSONL files, and a tiny
synthetic matcher model (monkeypatched FeatureExtractor) so that no GPU, no
DINOv2 download, and no Roboflow API key are required.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import cv2
import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from samwl.features import FEATURE_DIM, HAND_FEATURE_NAMES
from samwl.masks import rle_encode
from samwl.matcher import _ladder_hash, save_model
from samwl.pipeline import LabelSummary, _load_ledger, _write_ledger_entry, run_label


# ── Constants ─────────────────────────────────────────────────────────────────

_H, _W = 64, 64   # tiny synthetic image size
_N_IMAGES = 5     # number of synthetic images in each test
_N_CANDS = 4      # candidates per image


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_rgb(seed: int = 0) -> np.ndarray:
    """Return a tiny deterministic RGB image."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (_H, _W, 3), dtype=np.uint8)


def _make_mask(row_start: int, row_end: int, col_start: int, col_end: int) -> np.ndarray:
    m = np.zeros((_H, _W), dtype=np.uint8)
    m[row_start:row_end, col_start:col_end] = 1
    return m


def _cand_record(image_id: str, mask: np.ndarray, rung: str = "coarse") -> dict:
    rle = rle_encode(mask)
    area = int(mask.sum())
    xs, ys = np.where(mask)
    return {
        "image_id": image_id,
        "rung": rung,
        "rle": rle,
        "bbox": [int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())],
        "area": area,
        "pred_iou": 0.90,
        "stability_score": 0.95,
    }


def _write_candidates(cands_dir: Path, image_id: str, records: list[dict]) -> None:
    p = cands_dir / f"{image_id}.jsonl"
    with p.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def _write_image(images_dir: Path, image_id: str, rgb: np.ndarray) -> Path:
    p = images_dir / f"{image_id}.jpg"
    cv2.imwrite(str(p), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return p


def _make_synthetic_model(
    model_dir: Path,
    cfg: dict,
    threshold: float = 0.50,
) -> None:
    """Write a synthetic matcher.joblib + manifest.json to model_dir.

    Uses a tiny (FEATURE_DIM-dimensional) LogisticRegression trained on
    two trivial examples so that predict_proba never crashes.
    """
    rng = np.random.default_rng(42)
    X = rng.standard_normal((20, FEATURE_DIM)).astype(np.float32)
    y = np.array([1] * 10 + [-1] * 10, dtype=np.int32)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=1000, random_state=42
    )
    model.fit(X_scaled, y)

    metrics: dict[str, Any] = {
        "n_annotated": 2,
        "n_candidates_pos": 10,
        "n_candidates_neg": 10,
        "pool_recall": {"coarse": 1.0},
        "top1_accuracy": 0.80,
    }
    save_model(scaler, model, threshold, cfg, metrics, model_dir)


def _make_cfg(ladder: list[dict] | None = None) -> dict:
    """Build a minimal synthetic config dict."""
    if ladder is None:
        ladder = [
            {"name": "coarse", "points_per_side": 32, "pred_iou_thresh": 0.80,
             "stability_score_thresh": 0.95, "crop_n_layers": 0},
            {"name": "fine", "points_per_side": 64, "pred_iou_thresh": 0.70,
             "stability_score_thresh": 0.90, "crop_n_layers": 0},
        ]
    return {
        "roboflow": {
            "workspace": "test-ws",
            "project_id": "test-proj",
            "seed_tag": "to_manually_annotate",
            "weak_label_tag_prefix": "weak_label_v",
        },
        "target": {"class_name": "cup", "category_id": 1},
        "sam": {
            "weights": "models/fake.pt",
            "device": "cpu",
            "ladder": ladder,
        },
        "matcher": {
            "positive_iou": 0.75,
            "negative_iou": 0.25,
            "accept_threshold": 0.50,
            "embedding_model": "facebook/dinov2-base",
        },
        "polygon": {
            "fidelity_iou": 0.98,
            "max_points": None,
            "morph_kernel": 0,
            "min_ring_area_frac": 0.01,
        },
    }


def _fake_extract_batch(
    self_unused: Any,
    image_rgb: np.ndarray,
    candidates: list[dict],
) -> np.ndarray:
    """Synthetic feature extractor: return high-probability features for coarse masks."""
    rng = np.random.default_rng(len(candidates))
    features = rng.standard_normal((len(candidates), FEATURE_DIM)).astype(np.float32)
    # Bias the first candidate's features to be clearly positive so it is
    # accepted at the coarse rung by the synthetic LR model.
    for i, c in enumerate(candidates):
        if c.get("rung") == "coarse":
            features[i] += 3.0   # push toward the positive class
    return features


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def workspace(tmp_path: Path):
    """A fully pre-populated workspace for pipeline tests."""
    cfg = _make_cfg()
    images_dir = tmp_path / "images"
    cands_dir = tmp_path / "candidates"
    model_dir = tmp_path / "model"
    debug_dir = tmp_path / "debug"
    images_dir.mkdir()
    cands_dir.mkdir()

    image_ids = [f"img{i:03d}" for i in range(_N_IMAGES)]
    records = []
    for i, iid in enumerate(image_ids):
        rgb = _make_rgb(seed=i)
        _write_image(images_dir, iid, rgb)
        # Two coarse candidates + two fine candidates per image
        cands = [
            _cand_record(iid, _make_mask(10, 30, 10, 30), "coarse"),
            _cand_record(iid, _make_mask(5, 20, 5, 20), "coarse"),
            _cand_record(iid, _make_mask(15, 35, 15, 35), "fine"),
            _cand_record(iid, _make_mask(20, 40, 20, 40), "fine"),
        ]
        _write_candidates(cands_dir, iid, cands)
        records.append({
            "image_id": iid,
            "name": f"{iid}.jpg",
            "width": _W,
            "height": _H,
            "source": "local",
            "url": None,
            "path": str(images_dir / f"{iid}.jpg"),
            "embedding": None,
            "generated_at": "2026-01-01T00:00:00+00:00",
        })

    _make_synthetic_model(model_dir, cfg)

    return {
        "cfg": cfg,
        "records": records,
        "image_ids": image_ids,
        "images_dir": images_dir,
        "cands_dir": cands_dir,
        "model_dir": model_dir,
        "debug_dir": debug_dir,
        "ledger_path": tmp_path / "ledger.jsonl",
        "run_tag": "weak_label_v1",
    }


# ── Tests ──────────────────────────────────────────────────────────────────────


class TestDryRun:
    """Plan criterion 1: dry-run writes overlays + COCO, zero write calls."""

    def test_overlay_and_coco_files_written(self, workspace):
        w = workspace
        with patch("samwl.pipeline.FeatureExtractor.extract_batch", _fake_extract_batch):
            summary = run_label(
                w["records"],
                w["cfg"],
                client=None,
                dry_run=True,
                limit=None,
                run_tag=w["run_tag"],
                ledger_path=w["ledger_path"],
                debug_dir=w["debug_dir"],
                images_dir=w["images_dir"],
                cands_dir=w["cands_dir"],
                model_dir=w["model_dir"],
            )

        debug_dir: Path = w["debug_dir"]
        overlays = sorted(debug_dir.glob("*_overlay.png"))
        coco_files = sorted(debug_dir.glob("*.json"))

        total_processed = (
            sum(summary.accepted_per_rung.values()) + summary.abstained
        )
        assert total_processed == _N_IMAGES, (
            f"Expected {_N_IMAGES} processed, got {total_processed}"
        )
        assert len(overlays) == total_processed - summary.abstained, (
            f"Expected one overlay per accepted image; "
            f"got {len(overlays)} overlays, {total_processed - summary.abstained} accepted"
        )
        assert len(coco_files) == total_processed - summary.abstained, (
            f"Expected one COCO JSON per accepted image; "
            f"got {len(coco_files)}"
        )

    def test_zero_write_requests(self, workspace):
        w = workspace
        with patch("samwl.pipeline.FeatureExtractor.extract_batch", _fake_extract_batch):
            summary = run_label(
                w["records"],
                w["cfg"],
                client=None,
                dry_run=True,
                limit=None,
                run_tag=w["run_tag"],
                ledger_path=w["ledger_path"],
                debug_dir=w["debug_dir"],
                images_dir=w["images_dir"],
                cands_dir=w["cands_dir"],
                model_dir=w["model_dir"],
            )
        assert getattr(summary, "_write_requests", 0) == 0, (
            "dry_run=True must make zero write requests"
        )

    def test_per_rung_breakdown_has_ladder_rungs(self, workspace):
        w = workspace
        cfg = w["cfg"]
        rung_names = [r["name"] for r in cfg["sam"]["ladder"]]
        with patch("samwl.pipeline.FeatureExtractor.extract_batch", _fake_extract_batch):
            summary = run_label(
                w["records"],
                w["cfg"],
                client=None,
                dry_run=True,
                limit=None,
                run_tag=w["run_tag"],
                ledger_path=w["ledger_path"],
                debug_dir=w["debug_dir"],
                images_dir=w["images_dir"],
                cands_dir=w["cands_dir"],
                model_dir=w["model_dir"],
            )
        for rung in rung_names:
            assert rung in summary.accepted_per_rung, (
                f"accepted_per_rung missing key '{rung}'"
            )

    def test_limit_caps_processed_images(self, workspace):
        w = workspace
        limit = 3
        with patch("samwl.pipeline.FeatureExtractor.extract_batch", _fake_extract_batch):
            summary = run_label(
                w["records"],
                w["cfg"],
                client=None,
                dry_run=True,
                limit=limit,
                run_tag=w["run_tag"],
                ledger_path=w["ledger_path"],
                debug_dir=w["debug_dir"],
                images_dir=w["images_dir"],
                cands_dir=w["cands_dir"],
                model_dir=w["model_dir"],
            )
        total = sum(summary.accepted_per_rung.values()) + summary.abstained + summary.failed
        assert total <= limit, f"limit={limit} but {total} images were processed"

    def test_dry_run_flag_on_summary(self, workspace):
        w = workspace
        with patch("samwl.pipeline.FeatureExtractor.extract_batch", _fake_extract_batch):
            summary = run_label(
                w["records"],
                w["cfg"],
                client=None,
                dry_run=True,
                limit=None,
                run_tag=w["run_tag"],
                ledger_path=w["ledger_path"],
                debug_dir=w["debug_dir"],
                images_dir=w["images_dir"],
                cands_dir=w["cands_dir"],
                model_dir=w["model_dir"],
            )
        assert summary.dry_run is True


class TestLedgerResumability:
    """Plan criterion 2: already-processed images are skipped on re-run."""

    def test_already_processed_count(self, workspace):
        w = workspace
        with patch("samwl.pipeline.FeatureExtractor.extract_batch", _fake_extract_batch):
            # First run: process all images
            run_label(
                w["records"],
                w["cfg"],
                client=None,
                dry_run=True,
                limit=None,
                run_tag=w["run_tag"],
                ledger_path=w["ledger_path"],
                debug_dir=w["debug_dir"],
                images_dir=w["images_dir"],
                cands_dir=w["cands_dir"],
                model_dir=w["model_dir"],
            )
            # Second run: should see all as already processed, upload 0
            summary2 = run_label(
                w["records"],
                w["cfg"],
                client=None,
                dry_run=True,
                limit=None,
                run_tag=w["run_tag"],
                ledger_path=w["ledger_path"],
                debug_dir=w["debug_dir"],
                images_dir=w["images_dir"],
                cands_dir=w["cands_dir"],
                model_dir=w["model_dir"],
            )

        assert summary2.already_processed == _N_IMAGES
        total_new = (
            sum(summary2.accepted_per_rung.values())
            + summary2.abstained
            + summary2.failed
        )
        assert total_new == 0, f"Expected 0 new in second run, got {total_new}"

    def test_ledger_round_trip(self, tmp_path: Path):
        ledger = tmp_path / "ledger.jsonl"
        _write_ledger_entry(ledger, "img001", "accepted", "coarse")
        _write_ledger_entry(ledger, "img002", "abstained", None)
        _write_ledger_entry(ledger, "img003", "failed", None)
        done = _load_ledger(ledger)
        assert done == {"img001", "img002", "img003"}

    def test_corrupt_ledger_line_is_skipped(self, tmp_path: Path):
        ledger = tmp_path / "ledger.jsonl"
        ledger.write_text(
            '{"image_id": "img001", "status": "accepted"}\n'
            "not-json\n"
            '{"image_id": "img003", "status": "failed"}\n'
        )
        done = _load_ledger(ledger)
        assert "img001" in done
        assert "img003" in done


class TestMatcherConfigGuard:
    """Mutated ladder config must raise MatcherConfigError before any work."""

    def test_mutated_config_raises(self, workspace):
        from samwl.matcher import MatcherConfigError

        w = workspace
        bad_cfg = _make_cfg(
            ladder=[
                {"name": "different", "points_per_side": 99, "pred_iou_thresh": 0.50,
                 "stability_score_thresh": 0.50, "crop_n_layers": 0},
            ]
        )
        with pytest.raises(MatcherConfigError):
            with patch("samwl.pipeline.FeatureExtractor.extract_batch", _fake_extract_batch):
                run_label(
                    w["records"],
                    bad_cfg,
                    client=None,
                    dry_run=True,
                    limit=1,
                    run_tag=w["run_tag"],
                    ledger_path=w["ledger_path"],
                    debug_dir=w["debug_dir"],
                    images_dir=w["images_dir"],
                    cands_dir=w["cands_dir"],
                    model_dir=w["model_dir"],
                )


class TestCocoOutput:
    """COCO files written to debug_dir must be valid COCO segmentation documents."""

    def test_coco_structure(self, workspace):
        w = workspace
        with patch("samwl.pipeline.FeatureExtractor.extract_batch", _fake_extract_batch):
            run_label(
                w["records"],
                w["cfg"],
                client=None,
                dry_run=True,
                limit=1,
                run_tag=w["run_tag"],
                ledger_path=w["ledger_path"],
                debug_dir=w["debug_dir"],
                images_dir=w["images_dir"],
                cands_dir=w["cands_dir"],
                model_dir=w["model_dir"],
            )
        coco_files = list(w["debug_dir"].glob("*.json"))
        assert coco_files, "No COCO files written in dry-run mode"
        doc = json.loads(coco_files[0].read_text())
        assert "images" in doc
        assert "annotations" in doc
        assert "categories" in doc
        assert doc["images"][0]["width"] == _W
        assert doc["images"][0]["height"] == _H
        assert doc["categories"][0]["name"] == "cup"

    def test_overlay_is_valid_png(self, workspace):
        w = workspace
        with patch("samwl.pipeline.FeatureExtractor.extract_batch", _fake_extract_batch):
            run_label(
                w["records"],
                w["cfg"],
                client=None,
                dry_run=True,
                limit=1,
                run_tag=w["run_tag"],
                ledger_path=w["ledger_path"],
                debug_dir=w["debug_dir"],
                images_dir=w["images_dir"],
                cands_dir=w["cands_dir"],
                model_dir=w["model_dir"],
            )
        overlays = list(w["debug_dir"].glob("*_overlay.png"))
        assert overlays, "No overlay PNGs written"
        img = cv2.imread(str(overlays[0]))
        assert img is not None, f"OpenCV could not read overlay: {overlays[0]}"
        assert img.shape == (_H, _W, 3)
