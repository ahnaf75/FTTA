"""Task configuration lookups.

This module keeps a lightweight fast path for the common FTTA datasets and
falls back to the full legacy task registry on demand.
"""

from dataclasses import dataclass
from typing import Any, Dict

from .features import FeatureList


@dataclass
class TaskConfig:
    # The data_source_cls instantiates the DataSource,
    # which fetches data and preprocesses it using a preprocess_fn.
    data_source_cls: Any
    # The feature_list describes the schema of the data *after* the
    # preprocess_fn is applied.
    feature_list: FeatureList


def _build_fast_task_registry() -> Dict[str, TaskConfig]:
    from .data_source import (
        ANESDataSource,
        AssistmentsDataSource,
        DiabetesReadmissionDataSource,
        HELOCDataSource,
    )
    from tableshift.datasets.anes import ANES_FEATURES
    from tableshift.datasets.assistments import ASSISTMENTS_FEATURES
    from tableshift.datasets.diabetes_readmission import DIABETES_READMISSION_FEATURES
    from tableshift.datasets.heloc import HELOC_FEATURES

    return {
        "anes": TaskConfig(ANESDataSource, ANES_FEATURES),
        "assistments": TaskConfig(AssistmentsDataSource, ASSISTMENTS_FEATURES),
        "diabetes_readmission": TaskConfig(
            DiabetesReadmissionDataSource,
            DIABETES_READMISSION_FEATURES,
        ),
        "heloc": TaskConfig(HELOCDataSource, HELOC_FEATURES),
    }


_FAST_TASK_REGISTRY = _build_fast_task_registry()
_LEGACY_TASK_REGISTRY = None


def _get_legacy_registry() -> Dict[str, TaskConfig]:
    global _LEGACY_TASK_REGISTRY
    if _LEGACY_TASK_REGISTRY is None:
        from .tasks_legacy import _TASK_REGISTRY as legacy_registry

        _LEGACY_TASK_REGISTRY = legacy_registry
    return _LEGACY_TASK_REGISTRY


def get_task_config(name: str) -> TaskConfig:
    if name in _FAST_TASK_REGISTRY:
        return _FAST_TASK_REGISTRY[name]

    legacy_registry = _get_legacy_registry()
    if name in legacy_registry:
        return legacy_registry[name]

    raise NotImplementedError(f"task {name} not implemented.")
