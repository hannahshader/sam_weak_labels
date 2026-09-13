"""Seed-frame selection for the SAM weak-labels pipeline.

Picks n_frames visually diverse images from the manifest using greedy
farthest-point sampling in the Roboflow 768-d embedding space.  Falls back
to seeded-random sampling with a loud warning when embeddings are absent.
"""
from __future__ import annotations

import random
from typing import Any

import numpy as np
from loguru import logger


def _farthest_point_sample(
    embeddings: np.ndarray,
    n: int,
    seed: int = 42,
) -> list[int]:
    """Greedy farthest-point sampling in L2 embedding space.

    Selects n row indices from embeddings such that the chosen points
    maximise the minimum pairwise distance.  The initial point is chosen
    uniformly at random with the given seed, then each subsequent point is
    the one farthest from the current selected set.

    Parameters
    ----------
    embeddings : float64 array, shape (N, D)
        One embedding vector per image.
    n : int
        Number of points to select.  Must be ≤ N.
    seed : int
        RNG seed for the initial random pick, making selection deterministic
        for a fixed manifest.

    Returns
    -------
    list[int]
        Row indices of selected embeddings, in the order they were chosen.
    """
    n_total = len(embeddings)
    if n >= n_total:
        return list(range(n_total))

    rng = random.Random(seed)
    selected: list[int] = [rng.randrange(n_total)]

    # Tracks the minimum squared L2 distance of each candidate to the
    # currently selected set.  Starts at +inf; updated incrementally.
    min_sq_dists = np.full(n_total, np.inf, dtype=np.float64)

    for _ in range(n - 1):
        last = embeddings[selected[-1]]
        diffs = embeddings - last          # (N, D)
        sq_dists = (diffs * diffs).sum(axis=1)
        np.minimum(min_sq_dists, sq_dists, out=min_sq_dists)
        # Exclude already-selected points from future picks.
        min_sq_dists[selected[-1]] = -1.0
        nxt = int(np.argmax(min_sq_dists))
        selected.append(nxt)
        min_sq_dists[nxt] = -1.0

    return selected


def select_frames(
    records: list[dict[str, Any]],
    n_frames: int,
    *,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Select n_frames diverse image records from the manifest.

    Uses farthest-point sampling in the 768-d Roboflow embedding space when
    embeddings are present.  Falls back to seeded-random sampling with a loud
    warning if the manifest carries no embeddings at all.  When only a subset
    of records have embeddings, sampling is restricted to that subset and a
    warning is emitted.

    Parameters
    ----------
    records :
        Full list of manifest records (from ``samwl.manifest.read``).
    n_frames :
        Target number of frames.  Capped at ``len(records)`` automatically.
    seed :
        RNG seed.  A fixed seed makes selection fully deterministic for a
        given manifest ordering.

    Returns
    -------
    list[dict]
        Selected manifest records, one per chosen image.

    Raises
    ------
    ValueError
        If records is empty.
    """
    if not records:
        raise ValueError("record list is empty — nothing to sample from")

    n = min(n_frames, len(records))

    has_emb = [
        bool(r.get("embedding")) and len(r["embedding"]) > 0
        for r in records
    ]
    n_with_emb = sum(has_emb)

    if n_with_emb == 0:
        logger.warning(
            "No embeddings found in the manifest. "
            "Falling back to seeded-random sampling. "
            "To restore diversity-optimised selection, rebuild the manifest "
            "(`python main.py manifest build`) or upgrade the Roboflow plan."
        )
        rng = random.Random(seed)
        indices = rng.sample(range(len(records)), n)
        return [records[i] for i in indices]

    if n_with_emb < len(records):
        missing = len(records) - n_with_emb
        logger.warning(
            f"{missing}/{len(records)} records are missing embeddings. "
            "Farthest-point sampling will consider only the "
            f"{n_with_emb} records that have embeddings."
        )
        eligible = [r for r, has in zip(records, has_emb) if has]
    else:
        eligible = records

    mat = np.array([r["embedding"] for r in eligible], dtype=np.float64)
    indices = _farthest_point_sample(mat, n, seed=seed)
    return [eligible[i] for i in indices]
