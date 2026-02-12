from typing import List


def get_dataset(experiment: str, cache_dir: str):
    from tableshift import get_dataset as _get_dataset

    return _get_dataset(experiment, cache_dir)


def list_experiments() -> List[str]:
    from tableshift.configs import EXPERIMENT_CONFIGS

    return sorted(EXPERIMENT_CONFIGS.keys())
