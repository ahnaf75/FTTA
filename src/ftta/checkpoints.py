from typing import Dict


def resolve_checkpoint_path(
    experiment: str,
    model: str,
    model_root: str,
    checkpoint_maps: Dict,
    model_overrides: Dict,
) -> str:
    exp_map = checkpoint_maps.get("checkpoint_roots", {})
    exp_subdir = exp_map.get(experiment)
    if exp_subdir is None:
        print("please check experiment name!")
        raise ValueError("please check experiment name!")

    model_subdir = model_overrides.get("checkpoint_subdir")
    if model_subdir is None:
        if model == "mlp":
            model_subdir = "mlp"
        elif model == "tabtransformer":
            model_subdir = "tabtrans"
        elif model == "ft_transformer":
            model_subdir = "fttrans"
        else:
            raise ValueError(f"unsupported checkpoint model {model}")

    return f"{model_root}/{exp_subdir}/{model_subdir}/checkpoint.pt"
