"""Mask-quality logistic-regression classifier: label, train, evaluate, persist.

Step 8 of the SAM weak-labels plan.

Per-candidate workflow
----------------------
1. Run the full SAM ladder over every annotated seed frame.
2. Label each candidate +1 (IoU ≥ positive_iou) or -1 (IoU ≤ negative_iou);
   skip the ambiguous band (negative_iou < IoU < positive_iou).
3. Build DINOv2 + hand-crafted features for all candidates.
4. Evaluate by leave-one-frame-out cross-validation (LOFO) — never a random
   split, which would leak candidates from the same frame across the fold
   boundary.
5. Compute pool_recall per ladder rung and top1_accuracy from LOFO.
6. Compute a precision/coverage curve over accept_threshold.
7. Train a final L2 logistic regression with class_weight="balanced" on
   standardised features across all annotated frames.
8. Persist model + manifest; label refuses to run if the live ladder config
   hash does not match the manifest.

Public API
----------
label_candidates(candidates, gt_mask, positive_iou, negative_iou)
    -> (labels, ious)

build_frame_record(image_id, candidates, gt_mask, features_all, positive_iou, negative_iou)
    -> _FrameRecord

compute_pool_recall(records, rung_names, positive_iou)
    -> dict[rung_name, fraction]

run_lofo_cv(records)
    -> list[np.ndarray]  per-frame positive-class probability arrays

top1_accuracy(lofo_scores, records, positive_iou)
    -> float

precision_coverage_curve(lofo_scores, records, positive_iou, thresholds)
    -> list[tuple[float, float, float]]

threshold_at_precision(curve, target_precision)
    -> float

train_final(records)
    -> (scaler, model)

save_model(scaler, model, threshold, cfg, metrics, model_dir)
    -> (joblib_path, manifest_path)

load_model(model_dir, live_cfg)
    -> (scaler, model, threshold, manifest)
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from samwl.masks import mask_iou, rle_decode


# ── Exception ─────────────────────────────────────────────────────────────────


class MatcherConfigError(RuntimeError):
    """Raised when the live ladder config does not match the trained model's."""


# ── Data structure ────────────────────────────────────────────────────────────


@dataclass(eq=False)
class _FrameRecord:
    """All per-frame data needed for training and evaluation."""

    image_id: str
    #: Features for ALL candidates from this frame, shape (M, FEATURE_DIM).
    all_features: np.ndarray
    #: IoU of every candidate with the GT mask, shape (M,) float32.
    all_ious: np.ndarray
    #: Ladder rung name for every candidate, length M.
    all_rungs: list[str]
    #: Boolean mask selecting non-ambiguous candidates (ious outside the band).
    train_mask: np.ndarray
    #: Labels for the non-ambiguous subset: +1 positive, -1 negative, shape (N,).
    train_labels: np.ndarray


# ── Labelling ─────────────────────────────────────────────────────────────────


def label_candidates(
    candidates: list[dict],
    gt_mask: np.ndarray,
    positive_iou: float,
    negative_iou: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Label each candidate relative to a ground-truth mask.

    Parameters
    ----------
    candidates:
        Candidate records as stored by ``samwl.candidates``.  Each must have
        a ``rle`` field compatible with ``samwl.masks.rle_decode``.
    gt_mask:
        H×W uint8 ground-truth mask (1 = inside object).
    positive_iou, negative_iou:
        IoU thresholds.  Candidates with IoU ≥ positive_iou are positive (+1);
        IoU ≤ negative_iou are negative (-1); the band between is ambiguous (0).

    Returns
    -------
    (labels, ious)
        ``labels`` is int32 with values +1, -1, or 0 (ambiguous).
        ``ious`` is float32 IoU of each candidate with the GT mask.
    """
    gt_bool = gt_mask.astype(bool)
    ious = np.array(
        [mask_iou(rle_decode(c["rle"]).astype(bool), gt_bool) for c in candidates],
        dtype=np.float32,
    )
    labels = np.where(
        ious >= positive_iou,
        1,
        np.where(ious <= negative_iou, -1, 0),
    ).astype(np.int32)
    return labels, ious


def build_frame_record(
    image_id: str,
    candidates: list[dict],
    gt_mask: np.ndarray,
    features_all: np.ndarray,
    positive_iou: float,
    negative_iou: float,
) -> _FrameRecord:
    """Build a ``_FrameRecord`` from pre-computed features and a GT mask.

    Parameters
    ----------
    image_id:
        Roboflow image ID (used for logging).
    candidates:
        All candidate records for this frame (every rung).
    gt_mask:
        H×W uint8 ground-truth mask.
    features_all:
        Float32 array of shape (M, FEATURE_DIM) in the same order as
        ``candidates``.
    positive_iou, negative_iou:
        Labelling thresholds forwarded to ``label_candidates``.
    """
    m = len(candidates)
    if features_all.shape[0] != m:
        raise ValueError(
            f"{image_id}: {m} candidates but features_all has "
            f"{features_all.shape[0]} rows"
        )

    labels, ious = label_candidates(candidates, gt_mask, positive_iou, negative_iou)
    all_rungs = [c["rung"] for c in candidates]

    train_mask = labels != 0      # exclude ambiguous band
    train_labels = labels[train_mask]

    n_pos = int((labels == 1).sum())
    n_neg = int((labels == -1).sum())
    n_amb = int((labels == 0).sum())
    logger.debug(
        f"{image_id}: {m} candidates → {n_pos} pos, {n_neg} neg, {n_amb} ambiguous"
    )

    return _FrameRecord(
        image_id=image_id,
        all_features=features_all.astype(np.float32),
        all_ious=ious,
        all_rungs=all_rungs,
        train_mask=train_mask,
        train_labels=train_labels,
    )


# ── Pool recall ───────────────────────────────────────────────────────────────


def compute_pool_recall(
    records: list[_FrameRecord],
    rung_names: list[str],
    positive_iou: float,
) -> dict[str, float]:
    """Fraction of frames with ≥1 positive candidate, cumulative per rung.

    For rung k the fraction counts frames where any candidate from rungs 0..k
    has IoU ≥ positive_iou.  This shows whether each successive rung adds
    recall (i.e. whether escalation is buying anything).

    Parameters
    ----------
    records:
        Per-frame data — all candidates, all rungs.
    rung_names:
        Ladder rung names in order (from ``cfg["sam"]["ladder"]``).
    positive_iou:
        IoU threshold for a candidate to count as a "good" match.
    """
    n = len(records)
    if n == 0:
        return {r: 0.0 for r in rung_names}

    result: dict[str, float] = {}
    for k, rung in enumerate(rung_names):
        rung_prefix = set(rung_names[: k + 1])
        hits = sum(
            1
            for rec in records
            if any(
                rec.all_ious[j] >= positive_iou
                for j, r in enumerate(rec.all_rungs)
                if r in rung_prefix
            )
        )
        result[rung] = hits / n
    return result


# ── LR helpers ────────────────────────────────────────────────────────────────


def _fit_lr(X: np.ndarray, y: np.ndarray) -> tuple[Any, Any]:
    """Standardise X and fit an L2 logistic regression.

    Both classes must be present in y.  Returns (scaler, model).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        random_state=42,
    )
    model.fit(X_scaled, y)
    return scaler, model


# ── Leave-one-frame-out CV ────────────────────────────────────────────────────


def run_lofo_cv(records: list[_FrameRecord]) -> list[np.ndarray]:
    """Leave-one-frame-out cross-validation.

    For each frame i: train on the non-ambiguous candidates of every other
    frame, then score ALL candidates of frame i (including ambiguous ones).

    Parameters
    ----------
    records:
        At least 2 ``_FrameRecord`` instances required.

    Returns
    -------
    list of (M_i,) float32 arrays — positive-class probability for every
    candidate in the held-out frame.  Folds where training degrades (empty
    training set or single class) receive a zero-score array; a warning is
    logged in those cases.

    Raises
    ------
    ValueError
        If fewer than 2 frames are provided.
    """
    n = len(records)
    if n < 2:
        raise ValueError(f"LOFO CV requires ≥ 2 annotated frames, got {n}")

    scores_per_frame: list[np.ndarray] = []
    for i, held_out in enumerate(records):
        m = len(held_out.all_ious)

        X_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []
        for j, rec in enumerate(records):
            if j == i:
                continue
            if rec.train_mask.any():
                X_parts.append(rec.all_features[rec.train_mask])
                y_parts.append(rec.train_labels)

        if not X_parts:
            logger.warning(
                f"LOFO fold {i} ({held_out.image_id}): no training data — "
                "assigning zero scores"
            )
            scores_per_frame.append(np.zeros(m, dtype=np.float32))
            continue

        X_train = np.vstack(X_parts)
        y_train = np.concatenate(y_parts)

        unique_classes = np.unique(y_train)
        if len(unique_classes) < 2:
            logger.warning(
                f"LOFO fold {i} ({held_out.image_id}): training set has only "
                f"class(es) {unique_classes.tolist()} — assigning zero scores"
            )
            scores_per_frame.append(np.zeros(m, dtype=np.float32))
            continue

        scaler, model = _fit_lr(X_train, y_train)
        X_val = scaler.transform(held_out.all_features)
        proba = model.predict_proba(X_val)
        pos_idx = list(model.classes_).index(1)
        scores_per_frame.append(proba[:, pos_idx].astype(np.float32))

    return scores_per_frame


# ── Evaluation metrics ────────────────────────────────────────────────────────


def top1_accuracy(
    lofo_scores: list[np.ndarray],
    records: list[_FrameRecord],
    positive_iou: float,
) -> float:
    """Fraction of frames (with ≥1 positive candidate) where top-scored is positive.

    Parameters
    ----------
    lofo_scores:
        Output of ``run_lofo_cv``.
    records:
        Matching ``_FrameRecord`` list.
    positive_iou:
        IoU threshold for a "good" candidate.

    Returns
    -------
    float in [0, 1].  Returns 0.0 when no frame has a positive candidate.
    """
    eligible = 0
    correct = 0
    for scores, rec in zip(lofo_scores, records):
        if not np.any(rec.all_ious >= positive_iou):
            continue
        eligible += 1
        if len(scores) == 0:
            continue
        top_idx = int(np.argmax(scores))
        if rec.all_ious[top_idx] >= positive_iou:
            correct += 1
    return correct / eligible if eligible > 0 else 0.0


def precision_coverage_curve(
    lofo_scores: list[np.ndarray],
    records: list[_FrameRecord],
    positive_iou: float,
    thresholds: np.ndarray | None = None,
) -> list[tuple[float, float, float]]:
    """Precision/coverage curve swept over accept_threshold.

    At threshold t:
    - A frame is "accepted" when its top-scored candidate has score ≥ t.
    - Precision = fraction of accepted frames where the top candidate is positive
      (IoU ≥ positive_iou).
    - Coverage = fraction of all frames that are accepted.

    Parameters
    ----------
    thresholds:
        1-D array swept from low to high (default: 0.05, 0.10, …, 0.95).

    Returns
    -------
    list of (threshold, precision, coverage) tuples.
    ``precision`` is NaN when no frame is accepted at that threshold.
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 1.0, 0.05)

    n = len(records)
    if n == 0:
        return [(float(t), float("nan"), 0.0) for t in thresholds]

    top_scores = np.array(
        [float(s.max()) if len(s) > 0 else 0.0 for s in lofo_scores],
        dtype=np.float64,
    )
    top_is_positive = np.array(
        [
            len(lofo_scores[i]) > 0
            and bool(records[i].all_ious[int(np.argmax(lofo_scores[i]))] >= positive_iou)
            for i in range(n)
        ],
        dtype=bool,
    )

    result: list[tuple[float, float, float]] = []
    for t in thresholds:
        accepted = top_scores >= t
        n_accepted = int(accepted.sum())
        if n_accepted == 0:
            result.append((float(t), float("nan"), 0.0))
        else:
            precision = float(top_is_positive[accepted].sum()) / n_accepted
            coverage = n_accepted / n
            result.append((float(t), precision, coverage))
    return result


def threshold_at_precision(
    curve: list[tuple[float, float, float]],
    target: float = 0.95,
) -> float:
    """Return the lowest threshold where precision ≥ target.

    Falls back to the highest-precision threshold when the target is
    unreachable (with a logged warning).  Returns 0.5 when the curve is empty
    or entirely NaN.
    """
    eligible = [(t, p, c) for t, p, c in curve if not math.isnan(p) and p >= target]
    if eligible:
        return eligible[0][0]

    valid = [(t, p, c) for t, p, c in curve if not math.isnan(p)]
    if valid:
        best_t, best_p, _ = max(valid, key=lambda x: x[1])
        logger.warning(
            f"No threshold reached precision ≥ {target:.0%}. "
            f"Best available: threshold={best_t:.2f}, precision={best_p:.2%}. "
            "Annotate more seed frames and re-run fit."
        )
        return best_t

    return 0.5


# ── Final model training ──────────────────────────────────────────────────────


def train_final(records: list[_FrameRecord]) -> tuple[Any, Any]:
    """Train the final model on all annotated frames.

    Returns (scaler, model).

    Raises
    ------
    ValueError
        If no non-ambiguous candidates exist or only one class is present.
    """
    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    for rec in records:
        if rec.train_mask.any():
            X_parts.append(rec.all_features[rec.train_mask])
            y_parts.append(rec.train_labels)

    if not X_parts:
        raise ValueError("No non-ambiguous candidates found across all annotated frames.")

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)

    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        raise ValueError(
            f"Training set contains only class(es) {unique_classes.tolist()}. "
            "Need both positive and negative examples — annotate more frames."
        )

    logger.info(
        f"Training final model on {X.shape[0]} candidates "
        f"({int((y == 1).sum())} pos, {int((y == -1).sum())} neg)"
    )
    return _fit_lr(X, y)


# ── Config hash ───────────────────────────────────────────────────────────────


def _ladder_hash(ladder_cfg: list[dict[str, Any]]) -> str:
    """SHA-256 of the canonical JSON representation of the ladder config."""
    s = json.dumps(ladder_cfg, sort_keys=True)
    return hashlib.sha256(s.encode()).hexdigest()


# ── Persist / load ────────────────────────────────────────────────────────────


def save_model(
    scaler: Any,
    model: Any,
    threshold: float,
    cfg: dict[str, Any],
    metrics: dict[str, Any],
    model_dir: Path,
) -> tuple[Path, Path]:
    """Persist the model bundle and a human-readable manifest.

    Writes
    ------
    ``matcher.joblib``
        sklearn scaler + model + threshold + feature metadata.
    ``manifest.json``
        Config hash, feature layout, metrics, and ladder config.

    Returns
    -------
    (joblib_path, manifest_path)
    """
    import joblib

    from samwl.features import DINO_DIM, FEATURE_DIM, HAND_FEATURE_NAMES

    model_dir.mkdir(parents=True, exist_ok=True)

    ladder = cfg["sam"]["ladder"]
    ladder_hash = _ladder_hash(ladder)
    matcher_cfg = cfg.get("matcher", {})
    dino_model = matcher_cfg.get("embedding_model", "facebook/dinov2-base")

    bundle: dict[str, Any] = {
        "scaler": scaler,
        "model": model,
        "threshold": float(threshold),
        "feature_dim": FEATURE_DIM,
        "dino_dim": DINO_DIM,
        "hand_feature_names": HAND_FEATURE_NAMES,
        "dino_model": dino_model,
        "ladder_config_hash": ladder_hash,
    }

    joblib_path = model_dir / "matcher.joblib"
    joblib.dump(bundle, joblib_path)
    logger.info(f"Saved model bundle → {joblib_path}")

    manifest: dict[str, Any] = {
        "feature_dim": FEATURE_DIM,
        "dino_dim": DINO_DIM,
        "hand_feature_names": list(HAND_FEATURE_NAMES),
        "dino_model": dino_model,
        "ladder_config": ladder,
        "ladder_config_hash": ladder_hash,
        "positive_iou": float(matcher_cfg.get("positive_iou", 0.75)),
        "negative_iou": float(matcher_cfg.get("negative_iou", 0.25)),
        "accept_threshold": float(threshold),
        "metrics": metrics,
    }

    manifest_path = model_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info(f"Saved manifest → {manifest_path}")

    return joblib_path, manifest_path


def load_model(
    model_dir: Path,
    live_cfg: dict[str, Any],
) -> tuple[Any, Any, float, dict[str, Any]]:
    """Load the matcher model and validate the ladder config hash.

    Parameters
    ----------
    model_dir:
        Directory containing ``matcher.joblib`` and ``manifest.json``.
    live_cfg:
        Currently active pipeline config (from ``samwl.config.load``).

    Returns
    -------
    (scaler, model, threshold, manifest)

    Raises
    ------
    MatcherConfigError
        If the live ladder config hash differs from the one stored at training
        time.  The ``label`` command must check this before scoring any image.
    FileNotFoundError
        If either model file is missing.
    """
    import joblib

    manifest_path = model_dir / "manifest.json"
    joblib_path = model_dir / "matcher.joblib"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Matcher manifest not found: {manifest_path}")
    if not joblib_path.exists():
        raise FileNotFoundError(f"Matcher bundle not found: {joblib_path}")

    manifest = json.loads(manifest_path.read_text())

    live_hash = _ladder_hash(live_cfg["sam"]["ladder"])
    if live_hash != manifest["ladder_config_hash"]:
        raise MatcherConfigError(
            "Ladder config has changed since the model was trained.\n"
            f"  Manifest hash : {manifest['ladder_config_hash']}\n"
            f"  Live hash     : {live_hash}\n"
            "Re-run `python main.py fit` to retrain the model."
        )

    bundle = joblib.load(joblib_path)
    return (
        bundle["scaler"],
        bundle["model"],
        float(bundle["threshold"]),
        manifest,
    )
