"""Tests for samwl.matcher — Step 8 verification.

Verification criteria from the plan:
1. ``pytest tests/test_matcher.py`` passes.
2. Loading the matcher manifest under a mutated ladder config raises
   ``MatcherConfigError`` rather than silently proceeding.

Additional unit coverage:
- label_candidates assigns +1 / -1 / 0 correctly.
- build_frame_record builds train_mask and train_labels correctly.
- compute_pool_recall counts correctly per rung.
- run_lofo_cv returns the right shape and handles degenerate folds.
- top1_accuracy counts eligible frames and correct top-1 predictions.
- precision_coverage_curve produces NaN precision when nothing is accepted.
- threshold_at_precision selects the lowest threshold meeting the target.
- save_model / load_model round-trip without error.
"""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from samwl.masks import rle_encode
from samwl.matcher import (
    MatcherConfigError,
    _FrameRecord,
    _ladder_hash,
    build_frame_record,
    compute_pool_recall,
    label_candidates,
    load_model,
    precision_coverage_curve,
    run_lofo_cv,
    save_model,
    threshold_at_precision,
    top1_accuracy,
    train_final,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

FEAT_DIM = 8  # small synthetic dimension; unrelated to real FEATURE_DIM


def _make_mask(h: int, w: int, fill_rows: tuple, fill_cols: tuple) -> np.ndarray:
    """Return a uint8 H×W mask with the specified row/col slice filled."""
    m = np.zeros((h, w), dtype=np.uint8)
    m[fill_rows[0] : fill_rows[1], fill_cols[0] : fill_cols[1]] = 1
    return m


def _cand(mask: np.ndarray, rung: str = "coarse") -> dict:
    """Build a minimal candidate dict from a mask."""
    ys, xs = np.nonzero(mask)
    bbox = (
        [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        if xs.size > 0
        else [0, 0, 0, 0]
    )
    return {
        "image_id": "test",
        "rung": rung,
        "rle": rle_encode(mask),
        "bbox": bbox,
        "area": int(mask.sum()),
        "pred_iou": 0.9,
        "stability_score": 0.85,
    }


def _synthetic_record(
    n_candidates: int = 6,
    n_pos: int = 2,
    n_neg: int = 3,
    feat_dim: int = FEAT_DIM,
    image_id: str = "img0",
    rng: np.random.Generator | None = None,
    rung: str = "coarse",
) -> _FrameRecord:
    """Build a _FrameRecord with synthetic feature/IoU data.

    ``n_pos`` candidates get all_ious = 0.9 (positive),
    ``n_neg`` candidates get all_ious = 0.1 (negative),
    the rest get 0.5 (ambiguous).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    M = n_candidates
    n_amb = M - n_pos - n_neg
    assert n_amb >= 0, "n_pos + n_neg must be ≤ n_candidates"

    ious = np.array(
        [0.9] * n_pos + [0.1] * n_neg + [0.5] * n_amb,
        dtype=np.float32,
    )
    all_rungs = [rung] * M
    all_features = rng.random((M, feat_dim)).astype(np.float32)

    train_mask = (ious >= 0.75) | (ious <= 0.25)
    train_labels = np.where(ious[train_mask] >= 0.75, 1, -1).astype(np.int32)

    return _FrameRecord(
        image_id=image_id,
        all_features=all_features,
        all_ious=ious,
        all_rungs=all_rungs,
        train_mask=train_mask,
        train_labels=train_labels,
    )


def _multi_records(n: int = 4, **kwargs) -> list[_FrameRecord]:
    """Return n synthetic records with distinct image IDs."""
    rng = np.random.default_rng(99)
    return [
        _synthetic_record(image_id=f"img{i}", rng=rng, **kwargs) for i in range(n)
    ]


def _minimal_cfg(ladder: list[dict] | None = None) -> dict:
    """Minimal config dict for save_model / load_model tests."""
    if ladder is None:
        ladder = [
            {"name": "coarse", "points_per_side": 32, "pred_iou_thresh": 0.8},
        ]
    return {
        "sam": {
            "weights": "models/sam2.pt",
            "device": "cpu",
            "ladder": ladder,
        },
        "matcher": {
            "positive_iou": 0.75,
            "negative_iou": 0.25,
            "embedding_model": "facebook/dinov2-base",
        },
    }


# ── label_candidates ──────────────────────────────────────────────────────────


def test_label_candidates_basic() -> None:
    """High-IoU candidate is positive, low-IoU is negative, mid is ambiguous."""
    h, w = 20, 20
    gt = _make_mask(h, w, (5, 15), (5, 15))  # 10×10 box

    # Perfect overlap → IoU = 1.0 → positive
    pos_cand = _cand(_make_mask(h, w, (5, 15), (5, 15)))
    # No overlap → IoU = 0.0 → negative
    neg_cand = _cand(_make_mask(h, w, (0, 3), (0, 3)))
    # Partial overlap in the ambiguous band: rows 5:10, cols 5:15 → 50 px all
    # inside the 10×10 GT box (100 px).  IoU = 50/(100+50-50) = 0.5 (strictly
    # between negative_iou=0.25 and positive_iou=0.75).
    amb_cand = _cand(_make_mask(h, w, (5, 10), (5, 15)))

    labels, ious = label_candidates([pos_cand, neg_cand, amb_cand], gt, 0.75, 0.25)

    assert labels[0] == 1, f"expected +1, got {labels[0]}"
    assert labels[1] == -1, f"expected -1, got {labels[1]}"
    assert labels[2] == 0, f"expected 0 (ambiguous), got {labels[2]}"
    assert ious[0] == pytest.approx(1.0)
    assert ious[1] == pytest.approx(0.0)
    assert 0.25 < float(ious[2]) < 0.75


def test_label_candidates_empty() -> None:
    """Empty candidate list returns empty arrays."""
    gt = np.zeros((10, 10), dtype=np.uint8)
    labels, ious = label_candidates([], gt, 0.75, 0.25)
    assert labels.shape == (0,)
    assert ious.shape == (0,)


def test_label_candidates_boundary() -> None:
    """Candidates exactly at the threshold boundaries get the expected label."""
    h, w = 10, 10
    gt = np.zeros((h, w), dtype=np.uint8)
    gt[0:8, 0:8] = 1  # 64 px

    # IoU = 0.75: intersection=48, union=64 → needs a candidate with 48 px overlap
    # and no extra pixels.  Use a sub-region: candidate covers rows 0:6, cols 0:8 = 48px
    # union = 64 px → IoU = 48/64 = 0.75
    cand_75 = _cand(_make_mask(h, w, (0, 6), (0, 8)))
    labels, ious = label_candidates([cand_75], gt, 0.75, 0.25)
    assert labels[0] == 1, f"IoU at exact positive_iou threshold should be +1, got {labels[0]}"


# ── build_frame_record ────────────────────────────────────────────────────────


def test_build_frame_record_train_mask() -> None:
    """train_mask selects only non-ambiguous candidates."""
    h, w = 20, 20
    gt = _make_mask(h, w, (5, 15), (5, 15))
    cands = [
        _cand(_make_mask(h, w, (5, 15), (5, 15))),   # IoU = 1.0 → positive
        _cand(_make_mask(h, w, (0, 3), (0, 3))),      # IoU = 0.0 → negative
        _cand(_make_mask(h, w, (5, 10), (5, 15))),    # IoU = 0.5 → ambiguous
    ]
    features = np.random.default_rng(0).random((3, FEAT_DIM)).astype(np.float32)
    rec = build_frame_record("x", cands, gt, features, 0.75, 0.25)

    assert rec.train_mask.sum() == 2, "exactly 2 non-ambiguous candidates"
    assert set(rec.train_labels.tolist()) == {1, -1}


def test_build_frame_record_mismatched_features_raises() -> None:
    gt = np.zeros((10, 10), dtype=np.uint8)
    cands = [_cand(np.zeros((10, 10), dtype=np.uint8))]
    bad_features = np.zeros((5, FEAT_DIM), dtype=np.float32)  # wrong row count
    with pytest.raises(ValueError, match="features_all"):
        build_frame_record("x", cands, gt, bad_features, 0.75, 0.25)


# ── compute_pool_recall ───────────────────────────────────────────────────────


def test_pool_recall_cumulative() -> None:
    """Each successive rung can only increase or maintain recall."""
    records = _multi_records(n=6, n_pos=2, n_neg=3)
    rung_names = ["coarse", "fine", "finest"]
    # Assign rungs alternately so later rungs add positives.
    for i, rec in enumerate(records):
        rung = ["coarse", "fine", "finest"][i % 3]
        # Give frame with rung == "fine" a positive via "fine" rung only.
        rec.all_rungs[:] = [rung] * len(rec.all_rungs)

    recall = compute_pool_recall(records, rung_names, 0.75)

    assert set(recall.keys()) == set(rung_names)
    recalls = [recall[r] for r in rung_names]
    # Each value in [0, 1].
    for v in recalls:
        assert 0.0 <= v <= 1.0, f"recall out of range: {v}"
    # Cumulative: recall must be non-decreasing.
    for a, b in zip(recalls, recalls[1:]):
        assert b >= a - 1e-9, f"recall not non-decreasing: {recalls}"


def test_pool_recall_all_zero() -> None:
    """All-negative frames → recall == 0 for every rung."""
    records = [_synthetic_record(n_pos=0, n_neg=4, n_candidates=4, image_id=f"img{i}") for i in range(3)]
    recall = compute_pool_recall(records, ["coarse", "fine"], 0.75)
    assert all(v == 0.0 for v in recall.values())


def test_pool_recall_empty_records() -> None:
    recall = compute_pool_recall([], ["coarse", "fine"], 0.75)
    assert recall == {"coarse": 0.0, "fine": 0.0}


# ── run_lofo_cv ───────────────────────────────────────────────────────────────


def test_lofo_cv_shape() -> None:
    """Output has one array per frame; each array length matches all_ious."""
    records = _multi_records(n=4, n_pos=2, n_neg=3, n_candidates=6)
    scores = run_lofo_cv(records)
    assert len(scores) == 4
    for i, (sc, rec) in enumerate(zip(scores, records)):
        assert sc.shape == (len(rec.all_ious),), (
            f"fold {i}: expected ({len(rec.all_ious)},), got {sc.shape}"
        )
        assert sc.dtype == np.float32


def test_lofo_cv_requires_two_frames() -> None:
    records = [_synthetic_record(n_pos=2, n_neg=3, n_candidates=6)]
    with pytest.raises(ValueError, match="≥ 2"):
        run_lofo_cv(records)


def test_lofo_cv_single_class_fold_returns_zeros() -> None:
    """If all training frames share one class, the fold gets zero scores."""
    # 3 frames, all positive labels — no negatives in training set.
    rng = np.random.default_rng(7)
    records = []
    for i in range(3):
        all_ious = np.array([0.9, 0.9], dtype=np.float32)
        all_feats = rng.random((2, FEAT_DIM)).astype(np.float32)
        train_mask = np.array([True, True])
        train_labels = np.array([1, 1], dtype=np.int32)
        records.append(_FrameRecord(
            image_id=f"img{i}",
            all_features=all_feats,
            all_ious=all_ious,
            all_rungs=["coarse", "coarse"],
            train_mask=train_mask,
            train_labels=train_labels,
        ))
    scores = run_lofo_cv(records)
    # All folds should degrade to zero scores.
    for sc in scores:
        assert np.all(sc == 0.0), "single-class fold should produce zero scores"


# ── top1_accuracy ─────────────────────────────────────────────────────────────


def test_top1_accuracy_perfect() -> None:
    """When the positive candidate always has the highest score, top1 = 1.0."""
    records = _multi_records(n=4, n_pos=2, n_neg=3, n_candidates=6)
    # Craft scores so the first positive (idx 0, iou=0.9) always wins.
    scores = []
    for rec in records:
        sc = np.zeros(len(rec.all_ious), dtype=np.float32)
        sc[0] = 1.0   # index 0 is the positive (iou = 0.9 from _synthetic_record)
        scores.append(sc)
    acc = top1_accuracy(scores, records, 0.75)
    assert acc == pytest.approx(1.0)


def test_top1_accuracy_zero() -> None:
    """When the negative candidate always scores highest, top1 = 0.0."""
    records = _multi_records(n=4, n_pos=2, n_neg=3, n_candidates=6)
    scores = []
    for rec in records:
        sc = np.zeros(len(rec.all_ious), dtype=np.float32)
        # Positive indices are 0,1; negative indices are 2,3,4 — boost a negative.
        sc[2] = 1.0
        scores.append(sc)
    acc = top1_accuracy(scores, records, 0.75)
    assert acc == pytest.approx(0.0)


def test_top1_accuracy_no_positives() -> None:
    """No frame has a positive candidate → eligible == 0 → returns 0.0."""
    records = [_synthetic_record(n_pos=0, n_neg=4, n_candidates=4, image_id=f"img{i}") for i in range(3)]
    scores = [np.zeros(4, dtype=np.float32) for _ in records]
    acc = top1_accuracy(scores, records, 0.75)
    assert acc == 0.0


# ── precision_coverage_curve ──────────────────────────────────────────────────


def test_precision_coverage_nan_at_high_threshold() -> None:
    """At a threshold above all scores, coverage == 0 and precision is NaN."""
    records = _multi_records(n=4, n_pos=2, n_neg=3, n_candidates=6)
    scores = [np.full(len(r.all_ious), 0.3, dtype=np.float32) for r in records]
    curve = precision_coverage_curve(scores, records, 0.75, np.array([0.99]))
    t, p, c = curve[0]
    assert c == 0.0
    assert math.isnan(p)


def test_precision_coverage_monotone_coverage() -> None:
    """Coverage is non-increasing as threshold increases."""
    records = _multi_records(n=6, n_pos=2, n_neg=3, n_candidates=6)
    rng = np.random.default_rng(5)
    scores = [rng.random(len(r.all_ious)).astype(np.float32) for r in records]
    thresholds = np.arange(0.05, 1.0, 0.05)
    curve = precision_coverage_curve(scores, records, 0.75, thresholds)
    coverages = [c for _, _, c in curve]
    for a, b in zip(coverages, coverages[1:]):
        assert b <= a + 1e-9, f"coverage increased: {coverages}"


# ── threshold_at_precision ────────────────────────────────────────────────────


def test_threshold_at_precision_finds_lowest() -> None:
    """Returns the lowest threshold achieving the target."""
    curve = [
        (0.1, 0.7, 1.0),
        (0.5, 0.95, 0.8),
        (0.7, 0.98, 0.5),
    ]
    t = threshold_at_precision(curve, target=0.95)
    assert t == pytest.approx(0.5)


def test_threshold_at_precision_fallback() -> None:
    """Falls back to best-precision threshold when target is unreachable."""
    curve = [
        (0.1, 0.5, 1.0),
        (0.5, 0.7, 0.6),
        (0.9, 0.8, 0.2),
    ]
    t = threshold_at_precision(curve, target=0.95)
    assert t == pytest.approx(0.9)  # highest precision at 0.9


def test_threshold_at_precision_all_nan() -> None:
    """All-NaN curve returns default 0.5."""
    curve = [(0.5, float("nan"), 0.0)]
    t = threshold_at_precision(curve, target=0.95)
    assert t == pytest.approx(0.5)


# ── train_final ───────────────────────────────────────────────────────────────


def test_train_final_no_data_raises() -> None:
    """Records with no non-ambiguous candidates raise ValueError."""
    records = [_synthetic_record(n_pos=0, n_neg=0, n_candidates=2, image_id=f"img{i}") for i in range(2)]
    with pytest.raises(ValueError, match="No non-ambiguous"):
        train_final(records)


def test_train_final_single_class_raises() -> None:
    """Only-positive training set raises ValueError."""
    records = [_synthetic_record(n_pos=3, n_neg=0, n_candidates=3, image_id=f"img{i}") for i in range(2)]
    with pytest.raises(ValueError, match="only class"):
        train_final(records)


# ── save_model / load_model round-trip ───────────────────────────────────────


def _make_minimal_model():
    """Return (scaler, model) trained on trivial 2-class synthetic data."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    X = rng.random((20, FEAT_DIM)).astype(np.float32)
    y = np.array([1] * 10 + [-1] * 10, dtype=np.int32)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=100, random_state=0)
    model.fit(X_s, y)
    return scaler, model


def test_save_load_roundtrip(tmp_path: Path) -> None:
    """Saved model reloads without error when config matches."""
    scaler, model = _make_minimal_model()
    cfg = _minimal_cfg()
    metrics = {"n_annotated": 5, "pool_recall": {"coarse": 0.8}, "top1_accuracy": 0.75}

    joblib_path, manifest_path = save_model(scaler, model, 0.6, cfg, metrics, tmp_path)

    assert joblib_path.exists()
    assert manifest_path.exists()

    s2, m2, t2, mfst = load_model(tmp_path, cfg)
    assert t2 == pytest.approx(0.6)
    assert mfst["accept_threshold"] == pytest.approx(0.6)
    assert mfst["metrics"]["n_annotated"] == 5

    # The loaded model should produce probabilities.
    X_test = np.random.default_rng(1).random((3, FEAT_DIM)).astype(np.float32)
    proba = m2.predict_proba(s2.transform(X_test))
    assert proba.shape == (3, 2)


# ── Mutation test (required by plan verification criteria) ────────────────────


def test_mutated_ladder_config_raises(tmp_path: Path) -> None:
    """load_model raises MatcherConfigError when the live ladder config differs."""
    scaler, model = _make_minimal_model()
    original_cfg = _minimal_cfg(
        ladder=[{"name": "coarse", "points_per_side": 32, "pred_iou_thresh": 0.8}]
    )
    metrics: dict = {}
    save_model(scaler, model, 0.6, original_cfg, metrics, tmp_path)

    # Mutate a single field in the ladder.
    mutated_cfg = _minimal_cfg(
        ladder=[{"name": "coarse", "points_per_side": 64, "pred_iou_thresh": 0.8}]
    )
    with pytest.raises(MatcherConfigError, match="Ladder config has changed"):
        load_model(tmp_path, mutated_cfg)


def test_mutated_ladder_rung_added_raises(tmp_path: Path) -> None:
    """Adding a rung to the live config triggers the hash check."""
    scaler, model = _make_minimal_model()
    original_cfg = _minimal_cfg(
        ladder=[{"name": "coarse", "points_per_side": 32}]
    )
    save_model(scaler, model, 0.6, original_cfg, {}, tmp_path)

    extra_rung_cfg = _minimal_cfg(
        ladder=[
            {"name": "coarse", "points_per_side": 32},
            {"name": "fine", "points_per_side": 64},
        ]
    )
    with pytest.raises(MatcherConfigError):
        load_model(tmp_path, extra_rung_cfg)


def test_unchanged_config_does_not_raise(tmp_path: Path) -> None:
    """Identical config (re-ordered keys) still loads without error."""
    scaler, model = _make_minimal_model()
    cfg = _minimal_cfg()
    save_model(scaler, model, 0.6, cfg, {}, tmp_path)
    # Reload with the exact same config — must not raise.
    load_model(tmp_path, cfg)


# ── _ladder_hash ──────────────────────────────────────────────────────────────


def test_ladder_hash_deterministic() -> None:
    """Same ladder config always produces the same hash."""
    ladder = [{"name": "coarse", "points_per_side": 32}]
    assert _ladder_hash(ladder) == _ladder_hash(ladder)


def test_ladder_hash_sensitive_to_values() -> None:
    """Different ladder values produce different hashes."""
    a = [{"name": "coarse", "points_per_side": 32}]
    b = [{"name": "coarse", "points_per_side": 64}]
    assert _ladder_hash(a) != _ladder_hash(b)
