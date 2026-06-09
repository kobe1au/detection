from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in (update or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(path: str | Path) -> dict[str, Any]:
    """Load config with base inheritance and friendly missing-file guidance."""
    path = Path(path)
    if not path.exists():
        example = path.parent / f"{path.stem}.example{path.suffix}"
        if example.exists():
            raise FileNotFoundError(
                f"Config not found: {path}\n"
                f"Copy {example.name} to {path.name} and update paths for your environment."
            )
        raise FileNotFoundError(f"Config not found: {path}")

    cfg = load_yaml(path)
    bases = cfg.pop("base", None) or cfg.pop("bases", None) or []
    if isinstance(bases, (str, Path)):
        bases = [bases]
    merged: dict[str, Any] = {}
    for base in bases:
        base_path = Path(base)
        if not base_path.is_absolute():
            base_path = path.parent / base_path
        merged = deep_update(merged, load_config(base_path))
    return deep_update(merged, cfg)
