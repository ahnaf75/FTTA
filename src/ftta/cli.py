import argparse

from .engine import run_baseline, run_model_train
from .config_loader import get_runtime_defaults, load_all_configs


def _build_shared_parser(default_model: str) -> argparse.ArgumentParser:
    runtime_defaults = get_runtime_defaults(load_all_configs())
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache_dir",
        default=runtime_defaults.get("cache_dir", "tmp"),
        help="Directory to cache raw data files to.",
    )
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
    parser.add_argument(
        "--tta_method",
        default=runtime_defaults.get("tta_method", "ftta"),
        help="Test-time adaptation method to apply on ood_test.",
    )
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=int(runtime_defaults.get("num_seeds", 1)),
        help="Number of seeds to evaluate for TTA adaptation.",
    )
    parser.add_argument(
        "--seed_start",
        type=int,
        default=int(runtime_defaults.get("seed_start", 0)),
        help="Starting seed value for repeated evaluation.",
    )
    parser.add_argument(
        "--tent_lr",
        type=float,
        default=float(runtime_defaults.get("tent_lr", 1e-3)),
        help="Learning rate used by TENT adaptation.",
    )
    parser.add_argument(
        "--tent_steps",
        type=int,
        default=int(runtime_defaults.get("tent_steps", 1)),
        help="Number of entropy-minimization updates per batch for TENT.",
    )
    parser.add_argument(
        "--tent_eps",
        type=float,
        default=float(runtime_defaults.get("tent_eps", 1e-8)),
        help="Epsilon used in TENT entropy computation for numerical stability.",
    )
    parser.add_argument(
        "--sar_lr",
        type=float,
        default=float(runtime_defaults.get("sar_lr", 1e-3)),
        help="Learning rate used by SAR adaptation.",
    )
    parser.add_argument(
        "--sar_steps",
        type=int,
        default=int(runtime_defaults.get("sar_steps", 1)),
        help="Number of adaptation updates per batch for SAR.",
    )
    parser.add_argument(
        "--sar_rho",
        type=float,
        default=float(runtime_defaults.get("sar_rho", 0.05)),
        help="SAM neighborhood size used by SAR.",
    )
    parser.add_argument(
        "--sar_entropy_margin",
        type=float,
        default=float(runtime_defaults.get("sar_entropy_margin", 0.4)),
        help="Entropy filter threshold for SAR reliability selection.",
    )
    parser.add_argument(
        "--eata_lr",
        type=float,
        default=float(runtime_defaults.get("eata_lr", 1e-3)),
        help="Learning rate used by EATA adaptation.",
    )
    parser.add_argument(
        "--eata_steps",
        type=int,
        default=int(runtime_defaults.get("eata_steps", 1)),
        help="Number of adaptation updates per batch for EATA.",
    )
    parser.add_argument(
        "--eata_entropy_margin",
        type=float,
        default=float(runtime_defaults.get("eata_entropy_margin", 0.4)),
        help="Entropy threshold for EATA sample filtering.",
    )
    parser.add_argument(
        "--eata_diversity_margin",
        type=float,
        default=float(runtime_defaults.get("eata_diversity_margin", 0.05)),
        help="Cosine-distance threshold for EATA diversity filtering.",
    )
    return parser


def run_model_train_cli(argv=None):
    parser = _build_shared_parser(default_model="mlp")
    args = parser.parse_args(argv)
    run_model_train(**vars(args))


def run_baseline_cli(argv=None):
    parser = _build_shared_parser(default_model="histgbm")
    args = parser.parse_args(argv)
    run_baseline(**vars(args))
