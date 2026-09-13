"""Ground-truth annotation ingestion for the SAM weak-labels pipeline.

Loads hand-drawn polygon annotations for the seed frames from two sources:

  (a) A Roboflow COCO-segmentation export (``--coco PATH``), parsed the way
      ``sam_propagate.load_anchors`` does: ``annotations[].segmentation`` flat
      rings are rasterised with ``cv2.fillPoly``.

  (b) Direct fetch of ``urls.annotation`` per image via the Roboflow
      image-details endpoint — attempted first when ``coco_path`` is None.
      The JSON payload must contain Roboflow native-format polygon annotations
      (``annotations[].points`` + ``type == "polygon"``).  If the payload
      exists but is not polygon-segmentation, ``GroundTruthError`` is raised
      with a message naming ``--coco`` as the remedy.

Public API
----------
load(image_ids, *, client, coco_path, dims_by_id, name_by_id)
    Returns ``dict[image_id -> GroundTruth]``.  Images with no annotation are
    absent from the result; callers inspect ``len(result)`` vs ``len(image_ids)``
    to determine coverage.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests
from loguru import logger


class GroundTruthError(RuntimeError):
    """Raised when annotation data cannot be loaded or interpreted as polygons."""


@dataclass
class GroundTruth:
    """Rasterised ground-truth annotation for a single seed frame."""

    image_id: str
    width: int
    height: int
    #: One array per polygon ring, each shaped (N, 2) int32 (x, y columns).
    polygons: list[np.ndarray] = field(repr=False)
    #: H×W uint8 union of all rasterised polygons.
    mask: np.ndarray = field(repr=False)
    pixel_area: int
    n_polygons: int


# ── Rasterisation helper ──────────────────────────────────────────────────────


def _rasterise_polygons(
    point_lists: list[list[tuple[float, float]]],
    width: int,
    height: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Rasterise polygon point-lists into a uint8 union mask.

    Parameters
    ----------
    point_lists:
        Each element is a list of (x, y) float pairs forming one polygon ring.
        Rings with fewer than 3 points are silently dropped.
    width, height:
        Image dimensions for the output mask.

    Returns
    -------
    (mask, arrays)
        ``mask`` is a ``(height, width)`` uint8 array where 1 = inside any ring.
        ``arrays`` is one int32 Nx2 array per rasterised ring.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    arrays: list[np.ndarray] = []
    for pts_raw in point_lists:
        pts = np.array(pts_raw, dtype=np.int32)
        if pts.shape[0] < 3:
            continue
        cv2.fillPoly(mask, [pts], 1)
        arrays.append(pts)
    return mask, arrays


# ── COCO source (a) ───────────────────────────────────────────────────────────


def _load_from_coco(
    coco_path: Path,
    image_ids: list[str],
    name_by_id: dict[str, str],
) -> dict[str, GroundTruth]:
    """Load ground-truth from a Roboflow COCO segmentation export.

    Matches Roboflow image IDs to COCO records via the image ``name`` field
    (looked up through ``name_by_id``).  Matching first tries the full name,
    then the stem without extension.

    Parameters
    ----------
    coco_path:
        Path to a COCO JSON export produced by Roboflow.
    image_ids:
        Roboflow image IDs whose annotations are wanted.
    name_by_id:
        Mapping from Roboflow image ID to Roboflow image name (from the
        manifest).  Used to match against ``images[].file_name`` in the COCO
        file.
    """
    raw = json.loads(coco_path.read_text())

    # Index COCO images by file_name, its stem, and extra.name (original name
    # before Roboflow rewrites it to {stem}_jpg.rf.{hash}.jpg).
    coco_by_name: dict[str, dict[str, Any]] = {}
    for im in raw.get("images", []):
        fname = im.get("file_name", "")
        coco_by_name[fname] = im
        stem = Path(fname).stem
        if stem not in coco_by_name:
            coco_by_name[stem] = im
        extra_name = im.get("extra", {}).get("name", "")
        if extra_name and extra_name not in coco_by_name:
            coco_by_name[extra_name] = im
        extra_stem = Path(extra_name).stem
        if extra_stem and extra_stem not in coco_by_name:
            coco_by_name[extra_stem] = im

    # Index annotations by COCO image id.
    anns_by_coco_id: dict[int, list[dict[str, Any]]] = {}
    for ann in raw.get("annotations", []):
        anns_by_coco_id.setdefault(ann["image_id"], []).append(ann)

    results: dict[str, GroundTruth] = {}
    for rf_id in image_ids:
        name = name_by_id.get(rf_id, rf_id)
        coco_im = coco_by_name.get(name) or coco_by_name.get(Path(name).stem)
        if coco_im is None:
            logger.debug(f"COCO: no match for {rf_id!r} (name={name!r})")
            continue

        coco_id = coco_im["id"]
        anns = anns_by_coco_id.get(coco_id, [])
        if not anns:
            continue

        w, h = coco_im["width"], coco_im["height"]
        all_pts: list[list[tuple[float, float]]] = []
        for ann in anns:
            for seg in ann.get("segmentation", []):
                if not seg:
                    continue
                pts = [(float(seg[i]), float(seg[i + 1])) for i in range(0, len(seg) - 1, 2)]
                if len(pts) >= 3:
                    all_pts.append(pts)

        if not all_pts:
            continue

        mask, arrays = _rasterise_polygons(all_pts, w, h)
        results[rf_id] = GroundTruth(
            image_id=rf_id,
            width=w,
            height=h,
            polygons=arrays,
            mask=mask,
            pixel_area=int(mask.sum()),
            n_polygons=len(arrays),
        )

    return results


# ── API source (b) ────────────────────────────────────────────────────────────


def _parse_roboflow_annotation(
    payload: dict[str, Any],
    image_id: str,
    width: int,
    height: int,
) -> GroundTruth | None:
    """Parse a Roboflow native per-image annotation JSON.

    Returns ``None`` for images that have no annotations.  Raises
    ``GroundTruthError`` if annotations exist but are not polygon-segmentation,
    naming ``--coco`` as the remedy.

    Roboflow native format::

        {
          "annotations": [
            {"label": "...", "type": "polygon",
             "points": [{"x": 10.0, "y": 20.0}, ...]},
            ...
          ]
        }
    """
    annotations = payload.get("annotations", [])
    if not annotations:
        return None

    polygon_anns = [
        a for a in annotations
        if a.get("type") in (None, "polygon") and a.get("points")
    ]
    non_polygon_anns = [
        a for a in annotations
        if a.get("type") not in (None, "polygon")
    ]

    if non_polygon_anns and not polygon_anns:
        bad_types = sorted({a.get("type") for a in non_polygon_anns})
        raise GroundTruthError(
            f"Image {image_id!r}: annotation payload contains non-polygon "
            f"type(s) {bad_types}.  Roboflow only exports polygon segmentation "
            "via the native annotation endpoint.  Re-run with "
            "`--coco PATH` pointing to a COCO-segmentation export to load "
            "ground truth from a file."
        )

    all_pts: list[list[tuple[float, float]]] = []
    for ann in polygon_anns:
        pts = [(float(p["x"]), float(p["y"])) for p in ann.get("points", [])]
        if len(pts) >= 3:
            all_pts.append(pts)

    if not all_pts:
        return None

    mask, arrays = _rasterise_polygons(all_pts, width, height)
    return GroundTruth(
        image_id=image_id,
        width=width,
        height=height,
        polygons=arrays,
        mask=mask,
        pixel_area=int(mask.sum()),
        n_polygons=len(arrays),
    )


def _load_from_api(
    client: Any,
    image_ids: list[str],
    dims_by_id: dict[str, tuple[int, int]],
) -> dict[str, GroundTruth]:
    """Fetch ``urls.annotation`` per image and parse Roboflow polygon annotations.

    Parameters
    ----------
    client:
        ``RoboflowClient`` instance.
    image_ids:
        Roboflow image IDs for the seed frames.
    dims_by_id:
        Mapping from image_id to ``(width, height)``, typically from the
        manifest (the Roboflow image-details endpoint does not return
        dimensions reliably).
    """
    results: dict[str, GroundTruth] = {}

    for image_id in image_ids:
        details = client.image_details(image_id)
        img = details.get("image", details)
        urls = img.get("urls") or {}
        annotation_url = urls.get("annotation")

        if not annotation_url:
            logger.debug(f"{image_id}: no urls.annotation — not yet annotated")
            continue

        try:
            resp = requests.get(annotation_url, timeout=30)
            resp.raise_for_status()
            payload = resp.json()
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                # 404 on annotation URL means no annotation exists yet.
                logger.debug(f"{image_id}: annotation URL returned 404 — not annotated")
                continue
            logger.warning(f"{image_id}: annotation URL fetch failed: {exc}")
            continue
        except Exception as exc:
            logger.warning(f"{image_id}: annotation URL fetch failed: {exc}")
            continue

        # Prefer manifest dimensions; fall back to payload's image block.
        if image_id in dims_by_id:
            w, h = dims_by_id[image_id]
        else:
            im_meta = payload.get("image") or {}
            w = im_meta.get("width")
            h = im_meta.get("height")
            if not (isinstance(w, int) and w > 0 and isinstance(h, int) and h > 0):
                logger.warning(
                    f"{image_id}: cannot determine dimensions (not in manifest "
                    "or annotation payload) — skipping"
                )
                continue

        # May raise GroundTruthError for non-polygon types.
        gt = _parse_roboflow_annotation(payload, image_id, w, h)
        if gt is not None:
            results[image_id] = gt

    return results


# ── Public entry point ────────────────────────────────────────────────────────


def load(
    image_ids: list[str],
    *,
    client: Any = None,
    coco_path: Path | None = None,
    dims_by_id: dict[str, tuple[int, int]] | None = None,
    name_by_id: dict[str, str] | None = None,
) -> dict[str, GroundTruth]:
    """Load ground-truth annotations for seed frames.

    Source selection
    ----------------
    * If ``coco_path`` is provided: parse the COCO export (source a).
    * Otherwise: fetch ``urls.annotation`` via the Roboflow API (source b).
      ``client`` is required in this case.

    Parameters
    ----------
    image_ids:
        Roboflow image IDs whose ground truth is wanted.
    client:
        ``RoboflowClient`` instance.  Required unless ``coco_path`` is given.
    coco_path:
        Path to a Roboflow COCO-segmentation export.  When given, the API is
        not contacted.
    dims_by_id:
        Mapping from image_id to ``(width, height)``, used by the API path to
        obtain image dimensions (the Roboflow image-details endpoint does not
        return them reliably).  Build this from the manifest:
        ``{r["image_id"]: (r["width"], r["height"]) for r in records}``.
    name_by_id:
        Mapping from image_id to Roboflow image name, used by the COCO path
        to match ``images[].file_name``.  Build from the manifest:
        ``{r["image_id"]: r["name"] for r in records}``.

    Returns
    -------
    dict[str, GroundTruth]
        Keys are Roboflow image IDs.  Images with no annotation (or not found
        in the COCO export) are absent.  Callers compare ``len(result)`` to
        ``len(image_ids)`` for coverage.

    Raises
    ------
    GroundTruthError
        If an annotation payload exists but is not polygon-segmentation.
        The message names ``--coco`` as the remedy.
    ValueError
        If both ``coco_path`` and ``client`` are None.
    """
    if not image_ids:
        return {}

    if coco_path is not None:
        return _load_from_coco(coco_path, image_ids, name_by_id or {})

    if client is None:
        raise ValueError(
            "Either client or coco_path must be provided to load ground truth."
        )

    return _load_from_api(client, image_ids, dims_by_id or {})
