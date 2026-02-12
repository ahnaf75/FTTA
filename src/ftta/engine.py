from typing import Optional

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


def run_model_train(experiment: str, cache_dir: str, model: str, debug: bool):
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

    estimator = build_estimator(model, config)

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

    para, _ = torch.load(checkpoint_path, map_location=map_location)
    print(estimator.load_state_dict(para))

    estimator = train_estimator(estimator, dset, config=config)
    print(type(estimator))
    _print_completion(estimator, dset)
