"""Local image manifest — the seam between discovery and download.

Built once via ``python main.py manifest build`` (producer B: Roboflow
annotation-batch / annotation-job admin listing + ``/images/:id`` enrichment).
Downstream steps read the manifest; they never call ``/search``.

Schema (one JSON object per line in ``data/manifest.jsonl``)::

    {
      "image_id": str,
      "name": str,
      "width": int,
      "height": int,
      "source": "url" | "local",
      "url": str | null,      # when source == "url"
      "path": str | null,     # when source == "local"
      "embedding": list | null,
      "generated_at": str     # ISO-8601, same value on every row of a build
    }
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests
from loguru import logger
from tqdm import tqdm

from samwl.config import REPO_ROOT

DEFAULT_MANIFEST_PATH = REPO_ROOT / "data" / "manifest.jsonl"


class ManifestError(ValueError):
    """Raised for invalid manifest records or I/O failures."""


def _validate_record(rec: dict[str, Any]) -> None:
    if not rec.get("image_id"):
        raise ManifestError(f"record missing image_id: {rec!r}")
    w, h = rec.get("width"), rec.get("height")
    if not isinstance(w, int) or not isinstance(h, int) or w <= 0 or h <= 0:
        raise ManifestError(
            f"record {rec['image_id']}: width/height must be positive ints, "
            f"got {w!r}x{h!r}"
        )
    source = rec.get("source")
    if source == "url":
        if not rec.get("url"):
            raise ManifestError(f"record {rec['image_id']}: source=url but no url")
        if rec.get("path"):
            raise ManifestError(
                f"record {rec['image_id']}: source=url must not set path"
            )
    elif source == "local":
        if not rec.get("path"):
            raise ManifestError(f"record {rec['image_id']}: source=local but no path")
        if rec.get("url"):
            raise ManifestError(
                f"record {rec['image_id']}: source=local must not set url"
            )
    else:
        raise ManifestError(f"record {rec['image_id']}: invalid source {source!r}")


def write(records: list[dict[str, Any]], path: Path | None = None) -> Path:
    """Atomically write a validated manifest (``.tmp`` → fsync → ``os.replace``)."""
    path = path or DEFAULT_MANIFEST_PATH
    if not records:
        raise ManifestError("refusing to write an empty manifest")

    ids = [r["image_id"] for r in records]
    if len(ids) != len(set(ids)):
        raise ManifestError("duplicate image_id values in manifest")

    for rec in records:
        _validate_record(rec)

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return path


def read(path: Path | str | None = None) -> list[dict[str, Any]]:
    """Read and validate every record from a manifest file."""
    path = Path(path) if path else DEFAULT_MANIFEST_PATH
    if not path.exists():
        raise ManifestError(
            f"Manifest not found: {path}\n"
            "Run `python main.py manifest build` first."
        )
    records: list[dict[str, Any]] = []
    with path.open() as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ManifestError(f"{path}:{line_no}: invalid JSON") from exc
            _validate_record(rec)
            records.append(rec)
    return records


def _dims_from_url(url: str) -> tuple[int, int]:
    """Download an image into memory and return (width, height)."""
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    arr = np.frombuffer(resp.content, dtype=np.uint8)
    im = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise ManifestError(f"OpenCV could not decode URL: {url[:80]}…")
    h, w = im.shape[:2]
    return int(w), int(h)


def _enrich_one(client: Any, image_id: str, generated_at: str) -> dict[str, Any]:
    details = client.image_details(image_id)
    img = details.get("image", details)
    url = (img.get("urls") or {}).get("original") or img.get("url") or ""
    if not url:
        raise ManifestError(f"{image_id}: no urls.original in image_details")
    name = img.get("name") or image_id
    embedding = img.get("embedding")
    w = img.get("width")
    h = img.get("height")
    if not (isinstance(w, int) and isinstance(h, int) and w > 0 and h > 0):
        w, h = _dims_from_url(url)
    return {
        "image_id": image_id,
        "name": name,
        "width": w,
        "height": h,
        "source": "url",
        "url": url,
        "path": None,
        "embedding": embedding,
        "generated_at": generated_at,
    }


def build_from_roboflow(
    client: Any,
    *,
    path: Path | None = None,
    workers: int = 8,
) -> Path:
    """Producer B: list IDs from annotation batches/jobs, enrich, write manifest."""
    image_ids = client.list_all_image_ids()
    if not image_ids:
        raise ManifestError(
            "No image IDs found via annotation-batches / annotation-jobs. "
            "Is the project empty?"
        )
    logger.info(f"Building manifest for {len(image_ids)} image(s)…")
    generated_at = datetime.now(timezone.utc).isoformat()

    records_by_id: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_enrich_one, client, iid, generated_at): iid
            for iid in image_ids
        }
        with tqdm(total=len(futures), unit="img", desc="Manifest", ncols=80) as bar:
            for fut in as_completed(futures):
                iid = futures[fut]
                try:
                    records_by_id[iid] = fut.result()
                except Exception as exc:
                    errors.append(f"{iid}: {exc}")
                    logger.error(f"manifest enrich failed for {iid}: {exc}")
                bar.update()

    if errors:
        raise ManifestError(
            f"{len(errors)} image(s) failed during manifest build:\n"
            + "\n".join(errors[:20])
        )

    # Preserve discovery order.
    records = [records_by_id[iid] for iid in image_ids]
    out = write(records, path)
    logger.info(f"Wrote {len(records)} records → {out}")
    return out
