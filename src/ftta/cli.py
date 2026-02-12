import argparse

from .engine import run_baseline, run_model_train


def _build_shared_parser(default_model: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="tmp", help="Directory to cache raw data files to.")
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help=(
            "Whether to run in debug mode. If True, various truncations/"
            "simplifications are performed to speed up experiment."
        ),
    )
    parser.add_argument(
        "--experiment",
        default="diabetes_readmission",
        help="Experiment to run. Overridden when debug=True.",
    )
    parser.add_argument("--model", default=default_model, help="model to use.")
    return parser


def run_model_train_cli(argv=None):
    parser = _build_shared_parser(default_model="mlp")
    args = parser.parse_args(argv)
    run_model_train(**vars(args))


def run_baseline_cli(argv=None):
    parser = _build_shared_parser(default_model="histgbm")
    args = parser.parse_args(argv)
    run_baseline(**vars(args))
