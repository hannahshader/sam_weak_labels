"""SAM2 model and generator construction.

``build_generators`` is the single seam: swap this function if a future SAM
variant ships a compatible ``.generate(rgb) -> [dicts]`` interface.

Each ladder rung maps to one ``SAM2AutomaticMaskGenerator`` with the rung's
tuning knobs applied over the shared model.  The model is loaded once and
shared across generators to avoid reloading weights.

Config shape expected under ``cfg["sam"]``::

    weights: path/to/sam2.1_hiera_large.pt
    cfg:     configs/sam2.1/sam2.1_hiera_l.yaml  # relative to sam2 package
    device:  cuda
    ladder:
      - name: coarse
        points_per_side: 32
        pred_iou_thresh: 0.80
        stability_score_thresh: 0.95
        crop_n_layers: 0
      - ...
"""
from __future__ import annotations

from typing import Any

from loguru import logger


# Keys accepted by SAM2AutomaticMaskGenerator.__init__ that may appear in a
# ladder rung dict.  Any key not in this set is silently dropped so that
# extra documentation fields (e.g. ``name``) don't cause a TypeError.
_GENERATOR_KWARGS = frozenset(
    {
        "points_per_side",
        "points_per_batch",
        "pred_iou_thresh",
        "stability_score_thresh",
        "stability_score_offset",
        "mask_threshold",
        "box_nms_thresh",
        "crop_n_layers",
        "crop_nms_thresh",
        "crop_overlap_ratio",
        "crop_n_points_downscale_factor",
        "min_mask_region_area",
        "output_mode",
        "use_m2m",
        "multimask_output",
    }
)


def build_generators(cfg: dict[str, Any]):
    """Load the SAM2 model and return one generator per ladder rung.

    Returns a list of ``(rung_name, generator)`` tuples in ladder order.
    Raises ``ImportError`` if sam2 is not installed, and ``FileNotFoundError``
    if the weights file is missing.
    """
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2

    sam_cfg = cfg["sam"]
    weights = sam_cfg["weights"]
    model_cfg = sam_cfg.get("cfg", "configs/sam2.1/sam2.1_hiera_l.yaml")
    device = sam_cfg.get("device", "cuda")

    logger.info(f"Loading SAM2 model: weights={weights} cfg={model_cfg} device={device}")
    model = build_sam2(model_cfg, weights, device=device)

    generators = []
    for rung in sam_cfg["ladder"]:
        name = rung.get("name", f"rung{len(generators)}")
        kwargs = {k: v for k, v in rung.items() if k in _GENERATOR_KWARGS}
        logger.debug(f"Building generator for rung '{name}': {kwargs}")
        gen = SAM2AutomaticMaskGenerator(model, **kwargs)
        generators.append((name, gen))

    logger.info(f"Built {len(generators)} generator(s): {[n for n, _ in generators]}")
    return generators
