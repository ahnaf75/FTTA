from pathlib import Path
from typing import Any, Dict

import yaml


_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def _read_yaml(filename: str) -> Dict[str, Any]:
    path = _CONFIG_DIR / filename
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data


def load_all_configs() -> Dict[str, Dict[str, Any]]:
    return {
        "defaults": _read_yaml("defaults.yaml"),
        "models": _read_yaml("models.yaml"),
        "experiments": _read_yaml("experiments.yaml"),
    }


def get_runtime_defaults(configs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    return configs.get("defaults", {}).get("runtime", {})


def get_model_overrides(configs: Dict[str, Dict[str, Any]], model: str) -> Dict[str, Any]:
    model_cfg = configs.get("models", {}).get("models", {})
    merged = dict(model_cfg.get("default", {}))
    merged.update(model_cfg.get(model, {}))
    return merged


def get_checkpoint_maps(configs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    return configs.get("experiments", {}).get("experiments", {})
