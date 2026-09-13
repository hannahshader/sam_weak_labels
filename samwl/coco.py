"""Single-image COCO segmentation document builder.

Step 9 of the SAM weak-labels plan.

The document shape produced here matches what ``cup_weak_labels.build_coco``
produces, which is known to be accepted by Roboflow's ``save_annotation``
endpoint.

Public API
----------
build_coco(image_id, image_name, width, height, rings, category_name,
           category_id) -> dict

The returned dict is intended to be serialised with ``json.dumps`` / written to
a temporary file, then passed to
``Project.save_annotation(image_id, path, is_prediction=True)``.
"""
from __future__ import annotations

import datetime

import cv2
import numpy as np


def build_coco(
    image_id: str,
    image_name: str,
    width: int,
    height: int,
    rings: list[np.ndarray],
    category_name: str,
    category_id: int = 1,
) -> dict:
    """Build a single-image COCO segmentation document.

    Parameters
    ----------
    image_id:
        Roboflow image ID string.  Stored in ``info.source_image_id`` for
        traceability; not used as the COCO image id (which is always 1 for
        single-image documents).
    image_name:
        File name written to ``images[0].file_name`` (e.g. ``"frame_042.jpg"``).
    width, height:
        Image dimensions in pixels.
    rings:
        List of ``(N, 2)`` int32 arrays of ``(x, y)`` vertices produced by
        :func:`samwl.polygons.polygonize`.  Rings with fewer than 3 points are
        silently skipped.
    category_name:
        Category label string (e.g. ``"cup"``).
    category_id:
        COCO category id.  Defaults to 1.

    Returns
    -------
    dict
        A COCO segmentation document with keys ``info``, ``licenses``,
        ``categories``, ``images``, and ``annotations``.
    """
    coco_image_id = 1  # single-image document; internal id is always 1

    annotations: list[dict] = []
    ann_id = 1
    for ring in rings:
        ring = np.asarray(ring, dtype=np.int32)
        if len(ring) < 3:
            continue

        # Flat [x1, y1, x2, y2, ...] list as required by the COCO spec.
        seg: list[float] = ring.ravel().tolist()

        # Bounding box in COCO format: [x_min, y_min, width, height]
        x, y, bw, bh = cv2.boundingRect(ring)

        # Polygon area via the shoelace formula (OpenCV convention).
        area = float(cv2.contourArea(ring))

        annotations.append(
            {
                "id": ann_id,
                "image_id": coco_image_id,
                "category_id": category_id,
                "segmentation": [seg],
                "area": area,
                "bbox": [x, y, bw, bh],
                "iscrowd": 0,
            }
        )
        ann_id += 1

    return {
        "info": {
            "description": "SAM weak label",
            "version": "1.0",
            "year": datetime.date.today().year,
            "contributor": "sam_weak_labels",
            "date_created": datetime.date.today().isoformat(),
            "source_image_id": image_id,
        },
        "licenses": [],
        "categories": [
            {
                "id": category_id,
                "name": category_name,
                "supercategory": "object",
            }
        ],
        "images": [
            {
                "id": coco_image_id,
                "file_name": image_name,
                "width": width,
                "height": height,
            }
        ],
        "annotations": annotations,
    }
