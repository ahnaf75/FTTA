import argparse
import logging
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score

from tableshift import get_dataset
from tableshift.models.training import train
from tableshift.models.utils import get_estimator
from tableshift.models.default_hparams import get_default_config

LOG_LEVEL = logging.DEBUG

logger = logging.getLogger()
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    level=LOG_LEVEL,
    datefmt='%Y-%m-%d %H:%M:%S')


CHECKPOINT_PATHS = {
    "anes": {
        "mlp": "anes/mlp/checkpoint.pt",
        "tabtransformer": "anes/tabtrans/checkpoint.pt",
        "ft_transformer": "anes/fttrans/checkpoint.pt",
    },
    "assistments": {
        "mlp": "assistments/mlp/checkpoint.pt",
        "tabtransformer": "assistments/tabtrans/checkpoint.pt",
        "ft_transformer": "assistments/fttrans/checkpoint.pt",
    },
    "heloc": {
        "mlp": "heloc/mlp/checkpoint.pt",
        "tabtransformer": "heloc/tabtrans/checkpoint.pt",
        "ft_transformer": "heloc/fttrans/checkpoint.pt",
    },
    "diabetes_readmission": {
        "mlp": "diabetes/mlp/checkpoint.pt",
        "tabtransformer": "diabetes/tabtrans/checkpoint.pt",
        "ft_transformer": "diabetes/fttrans/checkpoint.pt",
    },
}


def load_checkpoint(estimator, experiment: str, model: str, models_dir: Path) -> None:
    if experiment not in CHECKPOINT_PATHS:
        raise ValueError(
            f"Unknown experiment '{experiment}'. Expected one of: "
            f"{', '.join(sorted(CHECKPOINT_PATHS.keys()))}."
        )
    if model not in CHECKPOINT_PATHS[experiment]:
        raise ValueError(
            f"Unknown model '{model}'. Expected one of: "
            f"{', '.join(sorted(CHECKPOINT_PATHS[experiment].keys()))}."
        )
    checkpoint_path = models_dir / CHECKPOINT_PATHS[experiment][model]
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. "
            "Make sure you have downloaded the pretrained weights."
        )
    parameters, _ = torch.load(checkpoint_path)
    print(estimator.load_state_dict(parameters))


def run_experiment(experiment: str, cache_dir: str, model: str, debug: bool) -> None:
    if debug:
        print("[INFO] running in debug mode.")
        experiment = "_debug"

    dset = get_dataset(experiment, cache_dir)
    _, _, _, _ = dset.get_pandas("train")
    config = get_default_config(model, dset)
    config["exp"] = experiment
    estimator = get_estimator(model, **config)

    if not debug:
        models_dir = Path(__file__).resolve().parent / "models"
        load_checkpoint(estimator, experiment, model, models_dir)

    estimator = train(estimator, dset, config=config)
    print(type(estimator))

    if not isinstance(estimator, torch.nn.Module):
        test_split = "ood_test" if dset.is_domain_split else "test"
        X_te, y_te, _, _ = dset.get_pandas(test_split)
        yhat_te = estimator.predict(X_te)
        acc = accuracy_score(y_true=y_te, y_pred=yhat_te)
        print(f"training completed! {test_split} accuracy: {acc:.4f}")
    else:
        print("training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache_dir",
        default="tmp",
        help="Directory to cache raw data files to.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help=(
            "Whether to run in debug mode. If True, various "
            "truncations/simplifications are performed to "
            "speed up experiment."
        ),
    )
    parser.add_argument(
        "--experiment",
        default="diabetes_readmission",
        help="Experiment to run. Overridden when debug=True.",
    )
    parser.add_argument("--model", default="mlp", help="model to use.")
    args = parser.parse_args()
    run_experiment(**vars(args))
