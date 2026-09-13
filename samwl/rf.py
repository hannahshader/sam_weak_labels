"""Roboflow read/write client for the SAM weak-labels pipeline.

Wraps the roboflow SDK for the project-handle lookup only; all enumeration and
mutation calls go through direct requests so we control status handling.

All public methods raise a named RFError subclass on failure and retry
transparently on 5xx / 429 with exponential backoff. They never return an empty
iterator or None to paper over a failure.

Note: the Roboflow /search endpoint is not available on all workspace plans.
When it returns 404 the client raises RFSearchUnavailableError (images exist
but cannot be enumerated via /search) or RFEmptyProjectError (no images).
"""
from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

import requests
from loguru import logger
from roboflow import Roboflow
from roboflow.config import API_URL

# ── Exceptions ───────────────────────────────────────────────────────────────


class RFError(RuntimeError):
    """Base class for Roboflow client errors."""


class RFClientError(RFError):
    """Raised for 4xx responses (bad request, not found, forbidden, …)."""

    def __init__(self, status: int, body: str, url: str) -> None:
        self.status = status
        self.body = body
        self.url = url
        super().__init__(f"HTTP {status} from {url}: {body[:200]}")


class RFEmptyProjectError(RFError):
    """Raised when the project has no images and /search returns 404."""

    def __init__(self, project_id: str) -> None:
        self.project_id = project_id
        super().__init__(
            f"Project '{project_id}' has no images. "
            "Upload images before running seed, fit, or label."
        )


class RFSearchUnavailableError(RFError):
    """Raised when /search returns 404 but the project contains images.

    This means the workspace plan does not expose the /search endpoint.
    Enumeration by image ID is not possible; batch-level counts are still
    available via get_batches() / unassigned_summary().
    """

    def __init__(self, project_id: str, image_count: int) -> None:
        self.project_id = project_id
        self.image_count = image_count
        super().__init__(
            f"Project '{project_id}' has {image_count} image(s) but the "
            "/search endpoint returned 404. The workspace plan may not include "
            "semantic search. Batch-level counts are available; per-image "
            "enumeration is not."
        )


class RFServerError(RFError):
    """Raised when retries are exhausted on 5xx / 429."""

    def __init__(self, status: int, url: str, attempts: int) -> None:
        self.status = status
        self.url = url
        self.attempts = attempts
        super().__init__(
            f"HTTP {status} from {url} after {attempts} attempt(s) — giving up"
        )


# ── Retry helper ─────────────────────────────────────────────────────────────

_RETRY_STATUSES = {429, 500, 502, 503, 504}
_MAX_ATTEMPTS = 5
_BASE_DELAY = 1.0  # seconds


def _request_with_retry(
    method: str,
    url: str,
    *,
    params: dict | None = None,
    json: Any = None,
    timeout: int = 30,
) -> requests.Response:
    """Execute an HTTP request, retrying on transient errors.

    Retries up to _MAX_ATTEMPTS times with exponential backoff on 429 and 5xx.
    Raises RFClientError immediately on 4xx. Raises RFServerError when retries
    are exhausted.
    """
    delay = _BASE_DELAY
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        resp = requests.request(
            method, url, params=params, json=json, timeout=timeout
        )
        if resp.ok:
            return resp
        if resp.status_code in _RETRY_STATUSES:
            if attempt == _MAX_ATTEMPTS:
                raise RFServerError(resp.status_code, url, attempt)
            wait = delay * (2 ** (attempt - 1))
            logger.warning(
                f"HTTP {resp.status_code} from {url} — retrying in {wait:.1f}s "
                f"(attempt {attempt}/{_MAX_ATTEMPTS})"
            )
            time.sleep(wait)
        else:
            raise RFClientError(resp.status_code, resp.text, url)
    raise RFServerError(-1, url, _MAX_ATTEMPTS)  # unreachable


# ── Client ───────────────────────────────────────────────────────────────────


class RoboflowClient:
    """Thin client over the Roboflow API for the SAM weak-labels pipeline.

    Parameters
    ----------
    cfg:
        Parsed config dict from ``samwl.config.load()``.  Only the
        ``roboflow.workspace`` and ``roboflow.project_id`` keys are used.
    api_key:
        Roboflow API key.  Obtain via ``samwl.config.require_api_key()``.
    """

    def __init__(self, cfg: dict, api_key: str) -> None:
        self._workspace = cfg["roboflow"]["workspace"]
        self._project_id = cfg["roboflow"]["project_id"]
        self._api_key = api_key

        # SDK used only to hold the project handle (save_annotation, etc.).
        # All enumeration goes through _search_page / get_batches directly.
        rf = Roboflow(api_key=api_key)
        self._project = rf.workspace(self._workspace).project(self._project_id)

    # ── private helpers ───────────────────────────────────────────────────────

    _PAGE_SIZE = 100
    _MAX_PAGES = 5_000  # safety cap: 500 000 images

    def _search_page(
        self,
        fields: list[str],
        tag: str | None,
        offset: int,
        limit: int,
    ) -> tuple[list[dict], int]:
        """Fetch one page from the /search endpoint.

        Returns ``(results, http_status)``.  A 404 is returned as
        ``([], 404)`` rather than raised so that callers can distinguish
        it from other 4xx errors and cross-reference /batches.  All other
        non-2xx statuses propagate as RFClientError / RFServerError.
        """
        url = f"{API_URL}/{self._workspace}/{self._project_id}/search"
        params: dict[str, Any] = {
            "api_key": self._api_key,
            "offset": offset,
            "limit": limit,
            "fields": ",".join(fields),
        }
        if tag is not None:
            params["tag"] = tag
        try:
            resp = _request_with_retry("GET", url, params=params)
            return resp.json().get("results", []), resp.status_code
        except RFClientError as exc:
            if exc.status == 404:
                return [], 404
            raise

    # ── public interface ──────────────────────────────────────────────────────

    def list_images(
        self,
        fields: list[str] | None = None,
        tag: str | None = None,
    ) -> Iterator[dict]:
        """Yield every image in the project (or filtered by tag).

        Parameters
        ----------
        fields:
            Roboflow image fields to include.  Defaults to ``["id", "name"]``.
            Available: id, name, created, annotations, labels, split, tags,
            owner, embedding, user_metadata.
        tag:
            If given, only yield images that have this tag.

        Yields
        ------
        dict
            One dict per image, containing the requested fields.

        Raises
        ------
        RFEmptyProjectError
            /search returned 404 and /batches confirms no images exist.
        RFSearchUnavailableError
            /search returned 404 but /batches reports images — the workspace
            plan does not include the search endpoint.
        """
        if fields is None:
            fields = ["id", "name"]

        offset = 0
        for _ in range(self._MAX_PAGES):
            page, status = self._search_page(fields, tag, offset, self._PAGE_SIZE)
            if status == 404:
                if offset == 0:
                    # First page 404: disambiguate via /batches.
                    batches = self.get_batches()
                    total = sum(b.get("images", 0) for b in batches)
                    if total == 0:
                        raise RFEmptyProjectError(self._project_id)
                    raise RFSearchUnavailableError(self._project_id, total)
                # 404 mid-pagination — treat defensively as end of results.
                logger.warning(
                    f"list_images: /search returned 404 at offset {offset}; "
                    "treating as end of results"
                )
                return
            yield from page
            if len(page) < self._PAGE_SIZE:
                return
            prev_offset = offset
            offset += self._PAGE_SIZE
            assert offset > prev_offset, "offset did not advance — possible API loop"

        raise RFError(
            f"list_images: exceeded {self._MAX_PAGES} pages — possible API loop"
        )

    def get_batches(self) -> list[dict]:
        """Return all upload batches for the project.

        Each dict contains at minimum: ``id``, ``name``, ``images``, and
        ``numJobs`` (the number of annotation jobs associated with the batch).

        Raises
        ------
        RFClientError
            On a 4xx response.
        RFServerError
            When retries are exhausted.
        """
        url = (
            f"{API_URL}/{self._workspace}/{self._project_id}/batches"
            f"?api_key={self._api_key}"
        )
        resp = _request_with_retry("GET", url)
        data = resp.json()
        return data.get("batches", [])

    def image_details(self, image_id: str) -> dict:
        """Return full metadata for a single image.

        The response includes ``id``, ``name``, ``urls`` (with ``original``),
        ``tags``, ``split``, and ``embedding`` when available.

        Raises
        ------
        RFClientError
            If the image is not found (404) or access is denied (403).
        """
        url = (
            f"{API_URL}/{self._workspace}/{self._project_id}/images/{image_id}"
            f"?api_key={self._api_key}"
        )
        resp = _request_with_retry("GET", url)
        return resp.json()

    def list_annotation_batches(self, *, show_empty: bool = False) -> list[dict]:
        """List annotation-board batches (admin API; not the same as /search)."""
        url = f"{API_URL}/{self._workspace}/{self._project_id}/annotation-batches"
        params: dict[str, Any] = {"api_key": self._api_key, "limit": 100}
        if show_empty:
            params["showEmpty"] = "true"
        resp = _request_with_retry("GET", url, params=params)
        return resp.json().get("batches", [])

    def list_jobs(self) -> list[dict]:
        """List annotation jobs via GET /jobs."""
        url = f"{API_URL}/{self._workspace}/{self._project_id}/jobs"
        resp = _request_with_retry("GET", url, params={"api_key": self._api_key})
        return resp.json().get("jobs", [])

    def _paginate_image_ids(self, path: str) -> list[str]:
        """Cursor-paginate an admin endpoint that returns ``imageIds``."""
        ids: list[str] = []
        after: str | None = None
        for _ in range(self._MAX_PAGES):
            params: dict[str, Any] = {
                "api_key": self._api_key,
                "limit": self._PAGE_SIZE,
            }
            if after:
                params["after"] = after
            resp = _request_with_retry("GET", path, params=params)
            body = resp.json()
            chunk = body.get("imageIds", [])
            ids.extend(chunk)
            after = body.get("nextPageToken")
            if not after or not chunk:
                break
        return ids

    def list_batch_image_ids(self, batch_id: str) -> list[str]:
        """List all image IDs in an annotation batch (paginated)."""
        path = (
            f"{API_URL}/{self._workspace}/{self._project_id}"
            f"/annotation-batches/{batch_id}/images"
        )
        return self._paginate_image_ids(path)

    def list_job_image_ids(self, job_id: str) -> list[str]:
        """List all image IDs in an annotation job (paginated)."""
        path = (
            f"{API_URL}/{self._workspace}/{self._project_id}"
            f"/annotation-jobs/{job_id}/images"
        )
        return self._paginate_image_ids(path)

    def list_all_image_ids(self) -> list[str]:
        """Union of image IDs across annotation batches and annotation jobs.

        Uses the admin listing endpoints that work without ``/search``.
        Emptied batches are included via ``showEmpty=true`` so images that
        have moved into jobs are not double-counted from a stale batch, while
        job listings still contribute IDs that left the upload batch.
        """
        seen: set[str] = set()
        ordered: list[str] = []
        for batch in self.list_annotation_batches(show_empty=True):
            for iid in self.list_batch_image_ids(batch["id"]):
                if iid not in seen:
                    seen.add(iid)
                    ordered.append(iid)
        for job in self.list_jobs():
            for iid in self.list_job_image_ids(job["id"]):
                if iid not in seen:
                    seen.add(iid)
                    ordered.append(iid)
        return ordered

    def add_tags(self, image_id: str, tags: list[str]) -> None:
        """Add one or more tags to an existing image.

        Uses the REST endpoint not exposed by the roboflow SDK:
        ``POST /:workspace/:project/images/:id/tags``
        with body ``{"operation": "add", "tags": [...]}``.

        Parameters
        ----------
        image_id:
            Roboflow image ID.
        tags:
            Tags to add.  Existing tags are not removed.

        Raises
        ------
        RFClientError
            On a 4xx response.
        """
        if not tags:
            return
        url = (
            f"{API_URL}/{self._workspace}/{self._project_id}"
            f"/images/{image_id}/tags?api_key={self._api_key}"
        )
        _request_with_retry("POST", url, json={"operation": "add", "tags": tags})

    def unassigned_summary(self) -> tuple[int, list[dict]]:
        """Return the unassigned image count and contributing batches.

        "Unassigned" means a batch with ``numJobs == 0``.  This method uses
        only ``get_batches()`` and makes no ``/search`` call, so it works
        regardless of whether the workspace plan includes search.

        Returns
        -------
        (total_unassigned, batches)
            ``total_unassigned`` is the sum of ``images`` over zero-job batches.
            ``batches`` is the list of those batch dicts from ``get_batches()``.
        """
        batches = self.get_batches()
        unassigned = [b for b in batches if b.get("numJobs", 0) == 0]
        total = sum(b.get("images", 0) for b in unassigned)
        return total, unassigned

    def save_weak_label(
        self,
        image_id: str,
        coco_path: Path,
        run_tag: str,
    ) -> None:
        """Upload a COCO prediction and apply the run tag.

        Calls the Roboflow SDK's ``save_annotation`` to route the image into
        the Review queue for its batch, then adds ``run_tag`` so the run is
        findable and revertible by tag filter.

        Parameters
        ----------
        image_id:
            Roboflow image ID.
        coco_path:
            Path to the COCO JSON file to upload.
        run_tag:
            Tag applied to the image after the annotation is saved
            (e.g. ``"weak_label_v1"``).

        Raises
        ------
        RFClientError
            On a 4xx response from the tag endpoint.
        RFServerError
            When retries are exhausted on 5xx / 429.
        """
        self._project.save_annotation(
            annotation_path=str(coco_path),
            image_id=image_id,
            is_prediction=True,
        )
        self.add_tags(image_id, [run_tag])

    def unassigned_image_ids(self) -> list[str]:
        """Return image IDs in upload batches with no annotation jobs.

        Requires ``/search`` to be available. Raises
        ``RFSearchUnavailableError`` when the endpoint returns 404.  Use
        ``unassigned_summary()`` when only the count is needed.

        Raises
        ------
        RFSearchUnavailableError
            When /search is not available on this workspace plan.
        RFClientError / RFServerError
            On other network failures.
        """
        batches = self.get_batches()
        unassigned_batch_ids = [
            b["id"] for b in batches if b.get("numJobs", 0) == 0
        ]
        if not unassigned_batch_ids:
            return []

        ids: list[str] = []
        for batch_id in unassigned_batch_ids:
            offset = 0
            for _ in range(self._MAX_PAGES):
                url = f"{API_URL}/{self._workspace}/{self._project_id}/search"
                params: dict[str, Any] = {
                    "api_key": self._api_key,
                    "offset": offset,
                    "limit": self._PAGE_SIZE,
                    "fields": "id",
                    "batch": "true",
                    "batch_id": batch_id,
                }
                try:
                    resp = _request_with_retry("GET", url, params=params)
                    page = resp.json().get("results", [])
                except RFClientError as exc:
                    if exc.status == 404:
                        # Cross-reference to give a useful error message.
                        total = sum(b.get("images", 0) for b in batches)
                        raise RFSearchUnavailableError(
                            self._project_id, total
                        ) from exc
                    raise
                ids.extend(img["id"] for img in page)
                if len(page) < self._PAGE_SIZE:
                    break
                offset += self._PAGE_SIZE
        return ids
