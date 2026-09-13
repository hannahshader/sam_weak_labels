"""Image cache for the SAM weak-labels pipeline.

Downloads full-resolution originals from Roboflow (urls.original) into
data/images/<image_id>.<ext>, skipping files that are already present and
valid. Every downloaded file is verified by decoding it with OpenCV and
checking that its pixel dimensions match Roboflow's reported metadata.

Entry point: ``fetch(items, cache_dir, workers)``
  items  — iterable of ``(image_id, url, expected_wh)`` triples
             where expected_wh is ``(width, height)`` from Roboflow or
             ``None`` to skip the dimension check.
  Returns a list of ``FetchResult`` namedtuples.

``fetch_from_client(image_ids, rf_client, cache_dir, workers)``
  Convenience wrapper: calls ``rf_client.image_details(id)`` per image to
  resolve the URL and expected dimensions, then calls ``fetch()``.
"""
from __future__ import annotations

import mimetypes
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import NamedTuple
from urllib.parse import urlparse

import cv2
import requests
from loguru import logger
from tqdm import tqdm

# ── Result type ───────────────────────────────────────────────────────────────


class FetchResult(NamedTuple):
    image_id: str
    path: Path
    width: int
    height: int
    bytes: int
    cached: bool  # True when the file was already present and valid


# ── Internal helpers ──────────────────────────────────────────────────────────

_CHUNK = 1 << 17  # 128 KiB


def _ext_from_url(url: str) -> str:
    """Derive a file extension from a URL; default to .jpg."""
    path = urlparse(url).path
    ext = Path(path).suffix.lower()
    if ext in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}:
        return ext
    # try content-type guessing via mimetypes
    mime = mimetypes.guess_type(path)[0] or ""
    guessed = mimetypes.guess_extension(mime) or ".jpg"
    return guessed


def _verify(path: Path, expected_wh: tuple[int, int] | None) -> tuple[int, int]:
    """Decode the file with OpenCV and return (width, height).

    Raises
    ------
    ValueError
        If the file cannot be decoded or its dimensions do not match
        ``expected_wh``.  The caller should delete the file and raise.
    """
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"OpenCV could not decode {path}")
    h, w = img.shape[:2]
    if expected_wh is not None:
        ew, eh = expected_wh
        if (w, h) != (ew, eh):
            raise ValueError(
                f"{path}: dimensions {w}x{h} do not match "
                f"Roboflow metadata {ew}x{eh}"
            )
    return w, h


def _download_one(
    image_id: str,
    url: str,
    expected_wh: tuple[int, int] | None,
    cache_dir: Path,
) -> FetchResult:
    """Download (or verify from cache) a single image."""
    ext = _ext_from_url(url)
    dest = cache_dir / f"{image_id}{ext}"

    # ── cache hit ────────────────────────────────────────────────────────────
    if dest.exists():
        try:
            w, h = _verify(dest, expected_wh)
            return FetchResult(
                image_id=image_id,
                path=dest,
                width=w,
                height=h,
                bytes=dest.stat().st_size,
                cached=True,
            )
        except ValueError as exc:
            logger.warning(f"Cached file invalid ({exc}); re-downloading.")
            dest.unlink(missing_ok=True)

    # ── download ─────────────────────────────────────────────────────────────
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
        with tmp.open("wb") as fh:
            for chunk in resp.iter_content(_CHUNK):
                fh.write(chunk)
        # verify before publishing
        w, h = _verify(tmp, expected_wh)
        tmp.rename(dest)
        return FetchResult(
            image_id=image_id,
            path=dest,
            width=w,
            height=h,
            bytes=dest.stat().st_size,
            cached=False,
        )
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


# ── Public API ────────────────────────────────────────────────────────────────


def fetch(
    items: list[tuple[str, str, tuple[int, int] | None]],
    cache_dir: Path,
    workers: int = 8,
) -> list[FetchResult]:
    """Download and cache a batch of images concurrently.

    Parameters
    ----------
    items:
        List of ``(image_id, url, expected_wh)`` triples.
        ``expected_wh`` may be ``None`` to skip the dimension check.
    cache_dir:
        Directory to store downloaded files.  Created if absent.
    workers:
        Maximum number of concurrent download threads.

    Returns
    -------
    list[FetchResult]
        One entry per input item, in the same order.  Raises on any
        per-image failure rather than returning a partial result.

    Raises
    ------
    RuntimeError
        If any image download or verification fails.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, FetchResult] = {}
    errors: list[str] = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_download_one, iid, url, wh, cache_dir): iid
            for iid, url, wh in items
        }
        with tqdm(
            total=len(futures),
            unit="img",
            desc="Fetching images",
            ncols=80,
        ) as bar:
            for fut in as_completed(futures):
                iid = futures[fut]
                try:
                    results[iid] = fut.result()
                except Exception as exc:
                    errors.append(f"{iid}: {exc}")
                    logger.error(f"Failed to fetch {iid}: {exc}")
                bar.update()

    if errors:
        raise RuntimeError(
            f"{len(errors)} image(s) failed to download:\n"
            + "\n".join(errors)
        )

    # return in original order
    return [results[iid] for iid, _, _ in items]


def fetch_from_client(
    image_ids: list[str],
    rf_client,
    cache_dir: Path,
    workers: int = 8,
) -> list[FetchResult]:
    """Resolve URLs via the Roboflow client then call ``fetch()``.

    Calls ``rf_client.image_details(id)`` for each ID to obtain
    ``urls.original`` and the reported image dimensions.  The dimension
    check uses ``width`` / ``height`` from the Roboflow response when
    present; it is skipped when absent.

    Parameters
    ----------
    image_ids:
        List of Roboflow image IDs to download.
    rf_client:
        An instance of ``samwl.rf.RoboflowClient``.
    cache_dir:
        Passed through to ``fetch()``.
    workers:
        Passed through to ``fetch()``.
    """
    items: list[tuple[str, str, tuple[int, int] | None]] = []
    for iid in image_ids:
        details = rf_client.image_details(iid)
        # The response shape is {"image": {...}} or the dict directly.
        img = details.get("image", details)
        url = img.get("urls", {}).get("original") or img.get("url", "")
        if not url:
            raise RuntimeError(
                f"Roboflow returned no urls.original for image {iid}. "
                "Response keys: " + str(list(img.keys()))
            )
        w = img.get("width")
        h = img.get("height")
        expected_wh = (int(w), int(h)) if w and h else None
        items.append((iid, url, expected_wh))

    return fetch(items, cache_dir, workers)


def fetch_from_manifest(
    records: list[dict],
    cache_dir: Path,
    workers: int = 8,
) -> list[FetchResult]:
    """Download or copy images described by manifest records.

    ``source == "url"`` downloads via ``fetch()``.
    ``source == "local"`` copies the file into the cache (verified the same way).
    """
    import shutil

    url_items: list[tuple[str, str, tuple[int, int] | None]] = []
    local_recs: list[dict] = []
    for rec in records:
        wh = (int(rec["width"]), int(rec["height"]))
        if rec["source"] == "url":
            url_items.append((rec["image_id"], rec["url"], wh))
        elif rec["source"] == "local":
            local_recs.append(rec)
        else:
            raise RuntimeError(f"Unknown manifest source: {rec.get('source')!r}")

    results_by_id: dict[str, FetchResult] = {}

    if url_items:
        for r in fetch(url_items, cache_dir, workers):
            results_by_id[r.image_id] = r

    for rec in local_recs:
        iid = rec["image_id"]
        src = Path(rec["path"])
        if not src.is_file():
            raise RuntimeError(f"Local manifest path missing: {src}")
        ext = src.suffix or ".jpg"
        dest = cache_dir / f"{iid}{ext}"
        cache_dir.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            try:
                w, h = _verify(dest, (int(rec["width"]), int(rec["height"])))
                results_by_id[iid] = FetchResult(
                    image_id=iid,
                    path=dest,
                    width=w,
                    height=h,
                    bytes=dest.stat().st_size,
                    cached=True,
                )
                continue
            except ValueError:
                dest.unlink(missing_ok=True)
        shutil.copy2(src, dest)
        w, h = _verify(dest, (int(rec["width"]), int(rec["height"])))
        results_by_id[iid] = FetchResult(
            image_id=iid,
            path=dest,
            width=w,
            height=h,
            bytes=dest.stat().st_size,
            cached=False,
        )

    return [results_by_id[rec["image_id"]] for rec in records]

