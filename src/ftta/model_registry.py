from typing import Any, Dict


def build_training_config(model: str, dset, model_overrides: Dict[str, Any]) -> Dict[str, Any]:
    from tableshift.models.default_hparams import get_default_config
    from tableshift.models.compat import is_pytorch_model_name

    config = get_default_config(model, dset)
    model_is_pt = is_pytorch_model_name(model)
    # Only keep override values that are true training/config keys.
    for key, value in model_overrides.items():
        if key in ("checkpoint_subdir",):
            continue
        if key == "tta_method" and not model_is_pt:
            continue
        config[key] = value
    return config


def build_estimator(model: str, config: Dict[str, Any]):
    from tableshift.models.utils import get_estimator

    return get_estimator(model, **config)


def train_estimator(estimator, dset, config: Dict[str, Any]):
    from tableshift.models.training import train

    return train(estimator, dset, config=config)
