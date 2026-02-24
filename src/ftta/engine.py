import random
from statistics import mean, pstdev
from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import accuracy_score

from .checkpoints import resolve_checkpoint_path
from .config_loader import (
    get_checkpoint_maps,
    get_model_overrides,
    get_runtime_defaults,
    load_all_configs,
)
from .data_registry import get_dataset
from .model_registry import build_estimator, build_training_config, train_estimator


def _print_completion(estimator, dset) -> None:
    if not isinstance(estimator, torch.nn.Module):
        test_split = "ood_test" if dset.is_domain_split else "test"
        X_te, y_te, _, _ = dset.get_pandas(test_split)
        yhat_te = estimator.predict(X_te)
        acc = accuracy_score(y_true=y_te, y_pred=yhat_te)
        print(f"training completed! {test_split} accuracy: {acc:.4f}")
    else:
        print("training completed!")


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _summarize_seeded_metrics(seed_metrics: List[Dict[str, float]], method_name: str) -> None:
    if len(seed_metrics) <= 1:
        return

    tracked_keys = [k for k in ("ood_test", "unadapt_ood_test") if k in seed_metrics[0]]
    if not tracked_keys:
        return

    print(f"{method_name.upper()} summary across {len(seed_metrics)} seeds:")
    for key in tracked_keys:
        values = [float(m[key]) for m in seed_metrics]
        std = pstdev(values) if len(values) > 1 else 0.0
        print(f"  {key}: mean={mean(values):.6f}, std={std:.6f}")


def run_baseline(experiment: str, cache_dir: str, model: str, debug: bool):
    configs = load_all_configs()
    runtime = get_runtime_defaults(configs)

    if debug:
        print("[INFO] running in debug mode.")
        experiment = runtime.get("debug_experiment", "_debug")

    if not cache_dir:
        cache_dir = runtime.get("cache_dir", "tmp")

    dset = get_dataset(experiment, cache_dir)
    _ = dset.get_pandas("train")

    model_overrides = get_model_overrides(configs, model)
    config = build_training_config(model, dset, model_overrides)
    estimator = build_estimator(model, config)
    estimator = train_estimator(estimator, dset, config=config)
    _print_completion(estimator, dset)


def run_model_train(
    experiment: str,
    cache_dir: str,
    model: str,
    debug: bool,
    tta_method: str = "ftta",
    num_seeds: int = 1,
    seed_start: int = 0,
    tent_lr: float = 1e-3,
    tent_steps: int = 1,
    tent_eps: float = 1e-8,
    sar_lr: float = 1e-3,
    sar_steps: int = 1,
    sar_rho: float = 0.05,
    sar_entropy_margin: float = 0.4,
    eata_lr: float = 1e-3,
    eata_steps: int = 1,
    eata_entropy_margin: float = 0.4,
    eata_diversity_margin: float = 0.05,
):
    configs = load_all_configs()
    runtime = get_runtime_defaults(configs)

    if debug:
        print("[INFO] running in debug mode.")
        experiment = runtime.get("debug_experiment", "_debug")

    if not cache_dir:
        cache_dir = runtime.get("cache_dir", "tmp")

    dset = get_dataset(experiment, cache_dir)
    _ = dset.get_pandas("train")

    model_overrides = get_model_overrides(configs, model)
    config = build_training_config(model, dset, model_overrides)
    config["exp"] = experiment
    config["tta_method"] = tta_method
    config["tent_lr"] = float(tent_lr)
    config["tent_steps"] = int(tent_steps)
    config["tent_eps"] = float(tent_eps)
    config["sar_lr"] = float(sar_lr)
    config["sar_steps"] = int(sar_steps)
    config["sar_rho"] = float(sar_rho)
    config["sar_entropy_margin"] = float(sar_entropy_margin)
    config["eata_lr"] = float(eata_lr)
    config["eata_steps"] = int(eata_steps)
    config["eata_entropy_margin"] = float(eata_entropy_margin)
    config["eata_diversity_margin"] = float(eata_diversity_margin)

    checkpoint_maps = get_checkpoint_maps(configs)
    checkpoint_path = resolve_checkpoint_path(
        experiment=experiment,
        model=model,
        model_root=runtime.get("model_root", "./models"),
        checkpoint_maps=checkpoint_maps,
        model_overrides=model_overrides,
    )

    map_location: Optional[torch.device] = None
    if not torch.cuda.is_available():
        map_location = torch.device("cpu")

    if num_seeds < 1:
        raise ValueError("num_seeds must be >= 1")

    seed_metrics: List[Dict[str, float]] = []
    estimator = None
    for seed in range(seed_start, seed_start + num_seeds):
        _set_global_seed(seed)
        seed_config = dict(config)
        seed_config["seed"] = seed
        estimator = build_estimator(model, seed_config)
        para, _ = torch.load(checkpoint_path, map_location=map_location)
        print(estimator.load_state_dict(para))

        fit_metrics = train_estimator(estimator, dset, config=seed_config)
        print(type(estimator))
        if isinstance(fit_metrics, dict):
            per_seed = {}
            for key, value in fit_metrics.items():
                if isinstance(value, list) and value:
                    per_seed[key] = float(value[-1])
                elif isinstance(value, (int, float)):
                    per_seed[key] = float(value)
            if per_seed:
                per_seed["seed"] = float(seed)
                seed_metrics.append(per_seed)

    _summarize_seeded_metrics(seed_metrics, tta_method)
    if estimator is not None:
        _print_completion(estimator, dset)
    return {
        "tta_method": tta_method,
        "seed_metrics": seed_metrics,
        "num_seeds": num_seeds,
        "seed_start": seed_start,
    }
