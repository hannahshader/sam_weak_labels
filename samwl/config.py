"""Load and validate pipeline configuration.

Reads config.yaml (or the path given in SAMWL_CONFIG) plus .env from the repo
root. All relative paths in the config are resolved against the repo root so
that the pipeline runs correctly regardless of the caller's cwd.

Raises ConfigError (a plain ValueError subclass) for every missing or invalid
field, rather than surfacing a KeyError or a cryptic AttributeError later.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent


class ConfigError(ValueError):
    """Raised for missing or invalid configuration."""


def _resolve(path_str: str) -> Path:
    """Resolve a config-relative path against the repo root."""
    p = Path(path_str)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def load(config_path: Path | None = None) -> dict[str, Any]:
    """Load, validate, and return the config as a plain dict.

    Side effect: loads .env from the repo root into the process environment.
    Raises ConfigError on any missing required field.
    """
    load_dotenv(REPO_ROOT / ".env")

    if config_path is None:
        env_override = os.environ.get("SAMWL_CONFIG")
        config_path = Path(env_override) if env_override else REPO_ROOT / "config.yaml"

    if not config_path.exists():
        raise ConfigError(
            f"Config file not found: {config_path}\n"
            "Copy config.example.yaml to config.yaml and fill in the values."
        )

    with config_path.open() as fh:
        cfg = yaml.safe_load(fh)

    if not isinstance(cfg, dict):
        raise ConfigError(f"Config file is empty or not a YAML mapping: {config_path}")

    _validate(cfg)
    _resolve_paths(cfg)
    return cfg


def _validate(cfg: dict) -> None:
    """Raise ConfigError for every missing required field."""
    required = {
        "roboflow.workspace",
        "roboflow.project_id",
        "target.class_name",
        "sam.weights",
        "sam.device",
        "sam.ladder",
    }
    missing = []
    for dotpath in required:
        parts = dotpath.split(".")
        node: Any = cfg
        for part in parts:
            if not isinstance(node, dict) or part not in node:
                missing.append(dotpath)
                break
            node = node[part]
        else:
            if node is None or node == "":
                missing.append(dotpath)

    if cfg.get("sam", {}).get("ladder") is not None:
        ladder = cfg["sam"]["ladder"]
        if not isinstance(ladder, list) or len(ladder) == 0:
            missing.append("sam.ladder (must be a non-empty list)")

    if missing:
        raise ConfigError(
            "Missing or empty required config fields:\n"
            + "\n".join(f"  - {f}" for f in missing)
        )


def require_api_key() -> str:
    """Return the Roboflow API key or raise ConfigError if absent."""
    key = os.environ.get("ROBOFLOW_API_KEY", "")
    if not key:
        raise ConfigError(
            "ROBOFLOW_API_KEY is not set. Add it to .env (see .env.example)."
        )
    return key


def _resolve_paths(cfg: dict) -> None:
    """Resolve relative paths inside the config dict in-place."""
    sam = cfg.setdefault("sam", {})
    if "weights" in sam:
        sam["weights"] = str(_resolve(sam["weights"]))
    # sam.cfg is relative to the sam2 package, not the repo root — leave it as-is.
