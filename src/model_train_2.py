import argparse
import copy
from torch.nn import functional as F
import logging
import math
import pickle
import json
import os
import numpy as np
from frozendict import frozendict
from ray.air.checkpoint import Checkpoint
from tableshift.third_party.domainbed import InfiniteDataLoader
from tableshift.models.rtdl import ResNetModel, MLPModel, FTTransformerModel
from sklearn.model_selection import train_test_split
from tableshift.core.utils import make_uid, convert_64bit_numeric_cols
import pandas as pd
from tableshift.models.default_hparams import _DEFAULT_CONFIGS
from pandas import DataFrame, Series
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, Union, Mapping, Sequence, Any, List, Tuple, Callable
import torch
from tableshift.core.features import Preprocessor, PreprocessorConfig, is_categorical
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from tableshift.datasets.mimic_extract import MIMIC_EXTRACT_STATIC_FEATURES
from tableshift.datasets.mimic_extract_feature_lists import \
    MIMIC_EXTRACT_SHARED_FEATURES
from tableshift.core.tasks import get_task_config, TaskConfig
from tableshift.models.torchutils import evaluate, evaluate_tta
from torch import nn
from tableshift.core.metrics import metrics_by_group

PYTORCH_DEFAULTS = frozendict({
    "lr": 0.001,
    "weight_decay": 0.0,
    "n_epochs": 1,
    "batch_size": 512,
})


DEFAULT_ID_TEST_SIZE = 0.1
DEFAULT_OOD_VAL_SIZE = 0.1
DEFAULT_ID_VAL_SIZE = 0.1
DEFAULT_RANDOM_STATE = 264738
GRINSTAJN_TEST_SIZE = 0.21
GRINSZTAJN_VAL_SIZE = 0.09

DEFAULT_BATCH_SIZE = 1024
OPTIMIZER_ARGS = ("lr", "weight_decay")

_MIMIC_EXTRACT_PASSTHROUGH_COLUMNS = [
    f for f in MIMIC_EXTRACT_SHARED_FEATURES.names
    if f not in MIMIC_EXTRACT_STATIC_FEATURES.names]

from tableshift.configs.non_benchmark_configs import NON_BENCHMARK_CONFIGS
@dataclass
class Grouper:
    features_and_values: Mapping[str, Sequence[Any]]
    drop: bool
    transformer: ColumnTransformer = None

    @property
    def features(self) -> List[str]:
        return list(self.features_and_values.keys())

    def _check_inputs(self, data: pd.DataFrame):
        """Check inputs to the transform function."""
        for c in self.features:
            assert c in data.columns, \
                f"data does not contain grouping feature {c}"
            data_vals = set(data[c].unique().tolist())
            group_vals = set(self.features_and_values[c])
            intersection = data_vals.intersection(group_vals)
            assert len(intersection), \
                f"None of the specified grouping values {group_vals} " \
                f"are in column {c} values {data_vals}. Do the grouping " \
                f"values have the same type as the column type {data[c].dtype}?"

            missing_ood_vals = set(group_vals) - set(data_vals)
            if len(missing_ood_vals):
                logging.warning(f"values {list(missing_ood_vals)} specified "
                      f"in Grouper split but not  present in the data.")
            return

    def _check_transformed(self, data: pd.DataFrame):
        """Check the outputs of the transform function."""
        for c in self.features:
            vals = data[c].unique()
            if len(vals) < 2:
                raise ValueError(f"[ERROR] column {c} contains only one "
                                 f"unique value after transformation {vals}")

        # Print a summary of the counts after grouping.
        logging.debug("overall counts after grouping:")
        if len(self.features) == 1:
            logging.info(data[self.features[0]].value_counts())
        else:
            row_feat = self.features[0]
            col_feats = self.features[1:]
            xt = pd.crosstab(data[row_feat].squeeze(),
                             data[col_feats].squeeze(),
                             dropna=False)
            logging.debug(xt)

    def _group_column(self, x: pd.Series, vals: Sequence) -> pd.Series:
        """Apply a grouping to a column, retuning a binary numeric Series."""
        # Ensure types are the same by casting vals to the same type as x.
        tmp = pd.Series(vals).astype(x.dtype)
        return x.isin(tmp).astype(int)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        self._check_inputs(data)
        for c in self.features:
            data[c] = self._group_column(data[c], self.features_and_values[c])
        self._check_transformed(data)
        return data

@dataclass
class PreprocessorConfig:
    # Preprocessing for categorical features (also applies to boolean features).
    # Options are: one_hot, map_values, label_encode, passthrough.
    categorical_features: str = "one_hot"
    # Preprocessing for float and int features.
    # Options: normalize, passthrough, map_values.
    numeric_features: str = "normalize"
    domain_labels: str = "label_encode"
    passthrough_columns: Union[
        str, List[str]] = None  # Feature names to passthrough, or "all".
    # If "rows", drop rows containing na values, if "columns", drop columns
    # containing na values; if None do not do anything for missing values.
    dropna: Union[str, None] = "rows"

    # If true, map feature name -> name_extended to features when name_extended
    # is specified.
    use_extended_names: bool = False

    # By defaults, target names/values are not mapped even when map_values is
    # used. However, if this is set to true, targets will be mapped when
    # map_values is specified.
    map_targets: bool = False

    default_targets_dtype = int
    cast_targets_to_default_type: bool = False

    min_frequency: float = None  # see OneHotEncoder.min_frequency
    max_categories: int = None  # see OneHotEncoder.max_categories
    n_bins: int = 5  # see KBinsDiscretizer.num_bins


def map_values(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    column = df.stack()
    unmapped_values = list(set(column.unique()) - set(mapping.keys()))
    if unmapped_values:
        logging.warning(
            f'got value(s) in column {df.columns[0]} with no'
            f'mapping: {unmapped_values}; will pass these values through '
            f'instead. If this is intended, then no action is required. '
            f'For guaranteed behavior (to ensure the value is not mapped), '
            f'it is best to define this explicitly in your mapping.')
        mapping.update({x: x for x in unmapped_values})

    return column.map(mapping).unstack()


def idx_where_in(x: pd.Series, vals: Sequence[Any]) -> np.ndarray:
    """Return a vector of the numeric indices i where X[i] in vals.

    Note that this function does not differentiate between different numeric
    types; i.e. if [1] (and integer) is in vals and x_j is 1.0 (a float),
     the jth index will be included in the output.
    """
    assert isinstance(vals, list) or isinstance(vals, tuple)
    idxs_bool = x.isin(vals)
    return np.nonzero(idxs_bool.values)[0]

def idx_where_not_in(x: pd.Series, vals: Sequence[Any]) -> np.ndarray:
    """Return a vector of the numeric indices i where X[i] not in vals.

    See note in idx_where_in regarding numeric types.
    """
    assert isinstance(vals, list) or isinstance(vals, tuple)
    idxs_bool = ~x.isin(vals)
    return np.nonzero(idxs_bool.values)[0]

@dataclass
class Splitter:
    """Splitter for non-domain splits."""
    val_size: float
    random_state: int

    @abstractmethod
    def __call__(self, data: pd.DataFrame, labels: pd.Series,
                 groups: pd.DataFrame = None, *args, **kwargs) -> Mapping[
        str, List[int]]:
        """Split a dataset.

        Returns a dictionary mapping split names to indices of the data points
        in that split."""
        raise


class FixedSplitter(Splitter):
    """A splitter for using fixed splits.

    This occurs, for example, when a dataset has a fixed train-test
    split (such as the Adult dataset).

    The FixedSplitter assumes there is a column in the dataset, "Split",
    which contains the values "train", "test".

    Note that for the fixed splitter, val_size indicates what fraction
    **of the training data** should be used for the validation set
    (since we cannot control the fraction of the overall data dedicated
    to validation, due to the prespecified train/test split).
    """

    def __init__(self, split_colname: str="Split", **kwargs):
        self.split_colname = split_colname
        super().__init__(**kwargs)

    def __call__(self, data: pd.DataFrame, labels: pd.Series,
                 groups: pd.DataFrame = None, *args, **kwargs) -> Mapping[
        str, List[int]]:

        assert self.split_colname in data.columns, "data is missing 'Split' column."
        assert all(np.isin(data[self.split_colname], ["train", "test"]))

        test_idxs = np.nonzero((data[self.split_colname] == "test").values)[0]
        train_val_idxs = \
            np.nonzero((data[self.split_colname] == "train").values)[0]

        train_idxs, val_idxs = train_test_split(
            train_val_idxs,
            train_size=(1 - self.val_size),
            random_state=self.random_state)

        del train_val_idxs
        return {"train": train_idxs, "validation": val_idxs, "test": test_idxs}

@dataclass
class RandomSplitter(Splitter):
    test_size: float

    @property
    def train_size(self):
        return 1. - (self.val_size + self.test_size)

    def __call__(self, data: pd.DataFrame, labels: pd.Series,
                 groups: pd.DataFrame = None, *args, **kwargs
                 ) -> Mapping[str, List[int]]:
        _check_input_indices(data)

        idxs = data.index.tolist()
        train_val_idxs, test_idxs = train_test_split(
            idxs,
            test_size=self.test_size,
            random_state=self.random_state)
        train_idxs, val_idxs = train_test_split(
            train_val_idxs,
            train_size=self.train_size / (self.train_size + self.val_size),
            random_state=self.random_state)
        del train_val_idxs
        return {"train": train_idxs, "validation": val_idxs, "test": test_idxs}

@dataclass
class DomainSplitter(Splitter):
    """Splitter for domain splits.

    All observations with domain_split_varname values in domain_split_ood_values
    are placed in the target (test) set; the remaining observations are split
    between the train, validation, and eval set.
    """
    id_test_size: float  # The in-domain test set.
    domain_split_varname: str

    domain_split_ood_values: Optional[Sequence[Any]] = None
    domain_split_id_values: Optional[Sequence[Any]] = None

    # If domain column is greater than this value, observation will be OOD.
    # If less than or equal to this value, observation will be ID.
    domain_split_gt_thresh: Optional[Union[int, float]] = None

    drop_domain_split_col: bool = True  # If True, drop column after splitting.
    ood_val_size: float = 0  # Fraction of OOD data to use for OOD validation set.

    def _split_from_explicit_values(self, domain_vals: pd.Series
                                    ) -> Tuple[np.ndarray, np.ndarray]:

        # Check that either in- or out-of-domain values are specified.
        assert self.is_explicit_split()

        # Check that threshold is not specified, since we are using the
        # explicit list of values to specify ID/OOD.
        assert self.domain_split_gt_thresh is None

        assert isinstance(self.domain_split_ood_values, tuple) \
               or isinstance(self.domain_split_ood_values, list), \
            "domain_split_ood_values must be an iterable type; got type {}".format(
                type(self.domain_split_ood_values))

        ood_vals = self.domain_split_ood_values

        # Fetch the out-of-domain indices.
        ood_idxs = idx_where_in(domain_vals, ood_vals)

        # Fetch the in-domain indices; these are either the explicitly-specified
        # in-domain values, or any values not in the OOD values.

        if self.domain_split_id_values is not None:
            # Check that there is no overlap between train/test domains.
            assert not set(self.domain_split_id_values).intersection(
                set(ood_vals))

            id_idxs = idx_where_in(domain_vals, self.domain_split_id_values)
            if not len(id_idxs):
                raise ValueError(
                    f"No ID observations with {self.domain_split_varname} "
                    f"values {self.domain_split_id_values}; are the values of "
                    f"same type as the column type of {domain_vals.dtype}?")
        else:
            id_idxs = idx_where_not_in(domain_vals, ood_vals)
            if not len(id_idxs):
                raise ValueError(
                    f"No ID observations with {self.domain_split_varname} "
                    f"values not in {ood_vals}.")

        if not len(ood_idxs):
            vals = domain_vals.unique()
            raise ValueError(
                f"No OOD observations with {self.domain_split_varname} values "
                f"{ood_vals}; are the values of same type"
                f" as the column type of {domain_vals.dtype}? Examples of "
                f"values in {self.domain_split_varname}: {vals[:10]}")

        return id_idxs, ood_idxs

    def _split_from_threshold(self, domain_vals: pd.Series) -> Tuple[
        np.ndarray, np.ndarray]:
        """Apply a threshold.

        Values are OOD if > self.domain_split_gt_thresh, else ID."""
        assert self.is_threshold_split()
        assert not self.is_explicit_split()

        if np.any(np.isnan(domain_vals)):
            logging.warning(
                f"detected missing values in domain column prior"
                "to splitting; this can result in unexpected behavior"
                "for threshold-based splits. Any nan values will"
                f"have OOD value: {np.nan > self.domain_split_gt_thresh}")

        ood_idxs = \
            np.nonzero((domain_vals > self.domain_split_gt_thresh).values)[0]
        id_idxs = \
            np.nonzero((domain_vals <= self.domain_split_gt_thresh).values)[0]
        return id_idxs, ood_idxs

    def is_explicit_split(self) -> bool:
        """Helper function to check whether an explicit split is used."""
        return (self.domain_split_ood_values is not None) or (
                self.domain_split_id_values is not None)

    def is_threshold_split(self) -> bool:
        """Helper function to check whether a threshold-based split is used."""
        return (self.domain_split_gt_thresh is not None)

    def __call__(self, data: pd.DataFrame, labels: pd.Series,
                 groups: pd.DataFrame = None, *args, **kwargs) -> Mapping[
        str, List[int]]:
        assert "domain_labels" in kwargs, "domain labels are required."
        domain_vals = kwargs.pop("domain_labels")
        assert isinstance(domain_vals, pd.Series)

        if self.is_explicit_split():
            id_idxs, ood_idxs = self._split_from_explicit_values(domain_vals)

        elif self.is_threshold_split():
            id_idxs, ood_idxs = self._split_from_threshold(domain_vals)

        else:
            raise NotImplementedError("Invalid domain split specified.")

        assert not set(id_idxs).intersection(ood_idxs), "sanity check for " \
                                                        "nonoverlapping " \
                                                        "domain split"
        assert not set(domain_vals.iloc[id_idxs]) \
            .intersection(domain_vals.iloc[ood_idxs]), "sanity check for no " \
                                                       "domain leakage"

        train_idxs, id_valid_eval_idxs = train_test_split(
            id_idxs, test_size=(self.val_size + self.id_test_size),
            random_state=self.random_state)

        valid_idxs, id_test_idxs = train_test_split(
            id_valid_eval_idxs,
            test_size=self.id_test_size / (self.val_size + self.id_test_size),
            random_state=self.random_state)

        outputs = {"train": train_idxs, "validation": valid_idxs,
                   "id_test": id_test_idxs}

        # Out-of-distribution splits
        if self.ood_val_size:
            ood_test_idxs, ood_valid_idxs = train_test_split(
                ood_idxs,
                test_size=self.ood_val_size,
                random_state=self.random_state)
            outputs["ood_test"] = ood_test_idxs
            outputs["ood_validation"] = ood_valid_idxs

        else:
            outputs["ood_test"] = ood_idxs

        return outputs


@dataclass
class ExperimentConfig:
    splitter: Splitter
    grouper: Union[Grouper, None]
    preprocessor_config: PreprocessorConfig
    tabular_dataset_kwargs: Dict[str, Any]
NON_BENCHMARK_CONFIGS = {
    "adult": ExperimentConfig(
        splitter=FixedSplitter(val_size=0.25, random_state=29746),
        grouper=Grouper({"Race": ["White", ], "Sex": ["Male", ]}, drop=False),
        preprocessor_config=PreprocessorConfig(), tabular_dataset_kwargs={}),

    "_debug": ExperimentConfig(
        splitter=DomainSplitter(
            val_size=0.01,
            id_test_size=0.2,
            ood_val_size=0.25,
            random_state=43406,
            domain_split_varname="purpose",
            # Counts by domain are below. We hold out all of the smallest
            # domains to avoid errors with very small domains during dev.
            # A48       9
            # A44      12
            # A410     12
            # A45      22
            # A46      50
            # A49      97
            # A41     103
            # A42     181
            # A40     234
            # A43     280
            domain_split_ood_values=["A44", "A410", "A45", "A46", "A48"]
        ),
        grouper=Grouper({"sex": ['1.0', ], "age_geq_median": ['1.0', ]},
                        drop=False),
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={"name": "german"}),

    "german": ExperimentConfig(
        splitter=RandomSplitter(val_size=0.01, test_size=0.2, random_state=832),
        grouper=Grouper({"sex": ['1.0', ], "age_geq_median": ['1.0', ]},
                        drop=False),
        preprocessor_config=PreprocessorConfig(), tabular_dataset_kwargs={}),

    "mooc": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="course_id",
                                domain_split_ood_values=[
                                    "HarvardX/CB22x/2013_Spring"]),
        grouper=Grouper({"gender": ["m", ],
                         "LoE_DI": ["Bachelor's", "Master's", "Doctorate"]},
                        drop=False),
        preprocessor_config=PreprocessorConfig(), tabular_dataset_kwargs={}),

    ################### Grinsztajn et al. benchmark datasets ###################

    **{x: ExperimentConfig(
        splitter=RandomSplitter(val_size=GRINSZTAJN_VAL_SIZE,
                                test_size=GRINSTAJN_TEST_SIZE,
                                random_state=DEFAULT_RANDOM_STATE),
        grouper=None,
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={"dataset_name": x}
    ) for x in ("electricity", "bank-marketing", "california",
                "covertype", "credit", 'default-of-credit-card-clients',
                'eye_movements', 'Higgs', 'MagicTelescope', 'MiniBooNE',
                'road-safety', 'pol', 'jannis', 'house_16H')},

    ################### MetaMIMIC datasets #####################################

    "metamimic_alcohol": ExperimentConfig(
        splitter=RandomSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                test_size=DEFAULT_ID_TEST_SIZE),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            numeric_features="kbins",
            passthrough_columns=["age"]
        ),
        tabular_dataset_kwargs={'name': 'metamimic_alcohol'}),

    'metamimic_anemia': ExperimentConfig(
        splitter=RandomSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                test_size=DEFAULT_ID_TEST_SIZE),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            numeric_features="kbins",
            passthrough_columns=["age"]
        ),
        tabular_dataset_kwargs={'name': 'metamimic_anemia'}),

    'metamimic_atrial': ExperimentConfig(
        splitter=RandomSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                test_size=DEFAULT_ID_TEST_SIZE),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            numeric_features="kbins",
            passthrough_columns=["age"]
        ),
        tabular_dataset_kwargs={'name': 'metamimic_atrial'}),

    'metamimic_diabetes': ExperimentConfig(
        splitter=RandomSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                test_size=DEFAULT_ID_TEST_SIZE),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            numeric_features="kbins",
            passthrough_columns=["age"]
        ),
        tabular_dataset_kwargs={'name': 'metamimic_diabetes'}),

    'metamimic_heart': ExperimentConfig(
        splitter=RandomSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                test_size=DEFAULT_ID_TEST_SIZE),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            numeric_features="kbins",
            passthrough_columns=["age"]
        ),
        tabular_dataset_kwargs={'name': 'metamimic_heart'}),

    'metamimic_hypertension': ExperimentConfig(
        splitter=RandomSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                test_size=DEFAULT_ID_TEST_SIZE),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            numeric_features="kbins",
            passthrough_columns=["age"]
        ),
        tabular_dataset_kwargs={'name': 'metamimic_hypertension'}),

    'metamimic_hypotension': ExperimentConfig(
        splitter=RandomSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                test_size=DEFAULT_ID_TEST_SIZE),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            numeric_features="kbins",
            passthrough_columns=["age"]
        ),
        tabular_dataset_kwargs={'name': 'metamimic_hypotension'}),

    'metamimic_ischematic': ExperimentConfig(
        splitter=RandomSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                test_size=DEFAULT_ID_TEST_SIZE),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            numeric_features="kbins",
            passthrough_columns=["age"]
        ),
        tabular_dataset_kwargs={'name': 'metamimic_ischematic'}),

    'metamimic_lipoid': ExperimentConfig(
        splitter=RandomSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                test_size=DEFAULT_ID_TEST_SIZE),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            numeric_features="kbins",
            passthrough_columns=["age"]
        ),
        tabular_dataset_kwargs={'name': 'metamimic_lipoid'}),

    'metamimic_overweight': ExperimentConfig(
        splitter=RandomSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                test_size=DEFAULT_ID_TEST_SIZE),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            numeric_features="kbins",
            passthrough_columns=["age"]
        ),
        tabular_dataset_kwargs={'name': 'metamimic_overweight'}),

    'metamimic_purpura': ExperimentConfig(
        splitter=RandomSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                test_size=DEFAULT_ID_TEST_SIZE),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            numeric_features="kbins",
            passthrough_columns=["age"]
        ),
        tabular_dataset_kwargs={'name': 'metamimic_purpura'}),

    'metamimic_respiratory': ExperimentConfig(
        splitter=RandomSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                test_size=DEFAULT_ID_TEST_SIZE),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            numeric_features="kbins",
            passthrough_columns=["age"]
        ),
        tabular_dataset_kwargs={'name': 'metamimic_respiratory'}),

    ################### CatBoost benchmark datasets ########################

    "amazon": ExperimentConfig(
        splitter=RandomSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                test_size=DEFAULT_ID_TEST_SIZE,
                                random_state=DEFAULT_RANDOM_STATE),
        grouper=None,
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={
            "kaggle_dataset_name": "amazon-employee-access-challenge"}),

    **{k: ExperimentConfig(
        splitter=RandomSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                test_size=DEFAULT_ID_TEST_SIZE,
                                random_state=DEFAULT_RANDOM_STATE),
        grouper=None,
        # categorical features in this dataset have *extremely* high cardinality
        preprocessor_config=PreprocessorConfig(
            categorical_features="passthrough"),
        tabular_dataset_kwargs={"task_name": k}) for k in
        ("appetency", "churn", "upselling")},

    "click": ExperimentConfig(
        splitter=FixedSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                               random_state=DEFAULT_RANDOM_STATE),
        grouper=None,
        # categorical features in this dataset have *extremely* high cardinality
        preprocessor_config=PreprocessorConfig(
            categorical_features="passthrough"),
        tabular_dataset_kwargs={
            "kaggle_dataset_name": "kddcup2012-track2"}),

    'kick': ExperimentConfig(
        splitter=RandomSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                test_size=DEFAULT_ID_TEST_SIZE,
                                random_state=DEFAULT_RANDOM_STATE),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            categorical_features="passthrough"),
        tabular_dataset_kwargs={"kaggle_dataset_name": "DontGetKicked"},
    ),

    ############# AutoML benchmark datasets (classification only) ##############
    **{x: ExperimentConfig(
        splitter=FixedSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                               random_state=DEFAULT_RANDOM_STATE),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            categorical_features="passthrough",
            dropna=None),
        tabular_dataset_kwargs={
            "automl_benchmark_dataset_name": x}) for x in (
        'product_sentiment_machine_hack', 'data_scientist_salary',
        'melbourne_airbnb', 'news_channel', 'wine_reviews',
        'imdb_genre_prediction', 'fake_job_postings2', 'kick_starter_funding',
        'jigsaw_unintended_bias100K',)},

    ######################## UCI datasets ######################################
    **{x: ExperimentConfig(
        splitter=RandomSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                test_size=DEFAULT_ID_TEST_SIZE,
                                random_state=DEFAULT_RANDOM_STATE),
        grouper=None,
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={}
    ) for x in ('iris', 'dry-bean', 'heart-disease', 'wine', 'wine-quality',
                'rice', 'cars', 'raisin', 'abalone')},

    # For breast cancer, mean/stc/worst values are already computed as features,
    # so we passthrough by default.
    'breast-cancer': ExperimentConfig(
        splitter=RandomSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                test_size=DEFAULT_ID_TEST_SIZE,
                                random_state=DEFAULT_RANDOM_STATE),
        grouper=None,
        preprocessor_config=PreprocessorConfig(numeric_features="passthrough"),
        tabular_dataset_kwargs={}),
    ######################## Kaggle datasets ###################################
    **{x: ExperimentConfig(
        splitter=RandomSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                test_size=DEFAULT_ID_TEST_SIZE,
                                random_state=DEFAULT_RANDOM_STATE),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            categorical_features="passthrough", dropna=None),
        tabular_dataset_kwargs={}
    ) for x in ('otto-products', 'sf-crime', 'plasticc', 'walmart',
                'tradeshift', 'schizophrenia', 'titanic',
                'santander-transactions', 'home-credit-default-risk',
                'ieee-fraud-detection', 'safe-driver-prediction',
                'santander-customer-satisfaction', 'amex-default',
                'ad-fraud')},

    ############################################################################

    "mimic_extract_los_3_selected": ExperimentConfig(
        splitter=RandomSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                test_size=DEFAULT_ID_TEST_SIZE,
                                random_state=DEFAULT_RANDOM_STATE),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            passthrough_columns=_MIMIC_EXTRACT_PASSTHROUGH_COLUMNS),
        tabular_dataset_kwargs={"task": "los_3",
                                "name": "mimic_extract_los_3_selected"}),

    "mimic_extract_mort_hosp_selected": ExperimentConfig(
        splitter=RandomSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                test_size=DEFAULT_ID_TEST_SIZE,
                                random_state=DEFAULT_RANDOM_STATE),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            passthrough_columns=_MIMIC_EXTRACT_PASSTHROUGH_COLUMNS),
        tabular_dataset_kwargs={"task": "mort_hosp",
                                "name": "mimic_extract_mort_hosp_selected"}),

    "communities_and_crime": ExperimentConfig(
        splitter=RandomSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                test_size=DEFAULT_ID_TEST_SIZE,
                                random_state=DEFAULT_RANDOM_STATE),
        grouper=None,
        preprocessor_config=PreprocessorConfig(), tabular_dataset_kwargs={}),

    "compas": ExperimentConfig(
        splitter=RandomSplitter(test_size=0.2, val_size=0.01,
                                random_state=90127),
        grouper=Grouper({"race": ["Caucasian", ], "sex": ["Male", ]},
                        drop=False),
        preprocessor_config=PreprocessorConfig(), tabular_dataset_kwargs={}),

}



##### IMPORTS ###########
# from tableshift import get_dataset
#### Stuff related to get_dataset


@dataclass
class PreprocessorConfig:
    # Preprocessing for categorical features (also applies to boolean features).
    # Options are: one_hot, map_values, label_encode, passthrough.
    categorical_features: str = "one_hot"
    # Preprocessing for float and int features.
    # Options: normalize, passthrough, map_values.
    numeric_features: str = "normalize"
    domain_labels: str = "label_encode"
    passthrough_columns: Union[
        str, List[str]] = None  # Feature names to passthrough, or "all".
    # If "rows", drop rows containing na values, if "columns", drop columns
    # containing na values; if None do not do anything for missing values.
    dropna: Union[str, None] = "rows"

    # If true, map feature name -> name_extended to features when name_extended
    # is specified.
    use_extended_names: bool = False

    # By defaults, target names/values are not mapped even when map_values is
    # used. However, if this is set to true, targets will be mapped when
    # map_values is specified.
    map_targets: bool = False

    default_targets_dtype = int
    cast_targets_to_default_type: bool = False

    min_frequency: float = None  # see OneHotEncoder.min_frequency
    max_categories: int = None  # see OneHotEncoder.max_categories
    n_bins: int = 5  # see KBinsDiscretizer.num_bins

@dataclass
class DatasetConfig:
    cache_dir: str = "tableshift_cache"
    download: bool = True
    random_seed: int = 948324


@dataclass
class Dataset(ABC):
    """Absract class to represent a dataset."""
    name: str
    preprocessor_config: PreprocessorConfig
    config: DatasetConfig
    initialize_data: bool
    splitter: Splitter = None
    splits = None  # dict mapping {split_name: list of idxs in split}
    grouper: Optional[Grouper] = None

    # List of the names of the predictors only.
    feature_names: Union[List[str], None] = None

    group_feature_names: Union[List[str], None] = None
    target: str = None

    domain_label_colname: Union[str, None] = None

    # If true, do not do per-domain evals
    skip_per_domain_eval: bool = False

    @property
    def cache_dir(self):
        return self.config.cache_dir

    @property
    def uid(self) -> str:
        return make_uid(self.name, self.splitter)

    @property
    def is_domain_split(self) -> bool:
        """Return True if this dataset uses a DomainSplitter, else False."""
        return isinstance(self.splitter, DomainSplitter)

    @property
    def eval_split_names(self) -> Tuple:
        """Fetch the names of the eval splits."""
        if self.skip_per_domain_eval:
            return tuple([x for x in self.splits if
                          x in ("test", "id_test", "ood_test")])

        else:
            return tuple([x for x in self.splits if "train" not in x])

    @property
    def domain_split_varname(self):
        if isinstance(self.splitter, DomainSplitter):
            return self.splitter.domain_split_varname
        else:
            return self.domain_label_colname

    @property
    @abstractmethod
    def n_domains(self) -> int:
        raise

    @property
    @abstractmethod
    def base_dir(self) -> str:
        "Return the location of the directory {cache_dir}/{uid}."
        raise

    @abstractmethod
    def _is_valid_split(self, split) -> bool:
        raise

    @abstractmethod
    def _initialize_data(self):
        """Load the data/labels/groups from a data source."""
        raise

    @property
    @abstractmethod
    def cat_idxs(self) -> List[int]:
        """Return a list of the indices of categorical columns."""
        raise

    @property
    @abstractmethod
    def features(self) -> List[str]:
        """Fetch a list of the feature names."""
        raise

    def _check_split(self, split):
        """Check that a split name is valid."""
        assert self._is_valid_split(split), \
            f"split {split} not in {list(self.splits.keys())}"

    @abstractmethod
    def _get_split_df(self, split: str, domain=None) -> pd.DataFrame:
        raise

    def _get_split_xygd(self, split, domain=None) -> Tuple[
        DataFrame, Series, DataFrame, Optional[Series]]:
        if domain is not None:
            raise NotImplementedError(
                "support for domain is not implemented in this class.")
        for name in ("feature_names", "target", "group_feature_names"):
            assert getattr(self, name) is not None, f"{name} is None."
        df = self._get_split_df(split, domain=domain)

        # preserve ordering of columns in df
        assert all(fn in df.columns for fn in self.feature_names)
        X = df[[c for c in df.columns if c in self.feature_names]]
        y = df[self.target]
        G = df[self.group_feature_names]
        d = df[self.domain_label_colname] \
            if self.domain_label_colname is not None else None
        return X, y, G, d

    def get_pandas(self, split, domain=None) -> Tuple[
        DataFrame, Series, DataFrame, Optional[Series]]:
        """Fetch the (data, labels, groups, domains) for this TabularDataset."""
        return self._get_split_xygd(split, domain)

    def get_dataloader(self, split, batch_size=2048,
                       shuffle=True, infinite=False) -> DataLoader:
        """Fetch a dataloader yielding (X, y, G, d) tuples."""
        data = self._get_split_xygd(split)

        if not self.domain_label_colname:
            # Drop the empty domain labels.
            data = data[:-1]
        return _make_dataloader_from_dataframes(data, batch_size, shuffle,
                                                infinite=infinite)

    def get_cache_dir(self, split: str, domain: Optional[Any] = None):
        if domain is None:
            return os.path.join(self.base_dir, split)
        else:
            return os.path.join(self.base_dir, split, str(domain))


def _make_dataloader_from_dataframes(
        data, batch_size: int, shuffle: bool,
        infinite=False) -> DataLoader:
    """Construct a (shuffled) DataLoader from a DataFrame."""
    data = tuple(map(lambda x: torch.tensor(x.values.astype(float)).float(), data))
    tds = torch.utils.data.TensorDataset(*data)
    if infinite:
        loader = InfiniteDataLoader(dataset=tds, batch_size=batch_size)
    else:
        loader = DataLoader(
            dataset=tds, batch_size=batch_size,
            shuffle=shuffle)
    return loader

class TabularDataset(Dataset):
    def __init__(self, name: str, config: DatasetConfig,
                 splitter: Splitter,
                 preprocessor_config: PreprocessorConfig,
                 grouper: Optional[Grouper],
                 initialize_data: bool = True,
                 task_config: Optional[TaskConfig] = None,
                 **kwargs):
        super().__init__(name=name,
                         preprocessor_config=preprocessor_config,
                         config=config,
                         grouper=grouper,
                         initialize_data=initialize_data,
                         splitter=splitter)

        # Dataset-specific info: features, data source, preprocessing.

        self.task_config = get_task_config(self.name) if task_config is None else task_config
        self.data_source = self.task_config.data_source_cls(
            cache_dir=self.config.cache_dir,
            download=self.config.download,
            **kwargs)

        self.preprocessor = Preprocessor(
            config=self.preprocessor_config,
            feature_list=self.task_config.feature_list)

        # Placeholders for data/labels/groups and split indices.
        self._df: Union[pd.DataFrame, None] = None  # holds all the data

        if initialize_data:
            self._initialize_data()

    @property
    def features(self) -> List[str]:
        return self.task_config.feature_list.names

    @property
    def predictors(self) -> List[str]:
        """The list of feature names in the FeatureList.

        Note that these do *not* necessarily correspond to the names in X,
        the data provided after preprocessing."""
        return self.task_config.feature_list.predictors

    @property
    def X_shape(self):
        """Shape of the data matrix for training."""
        return [None, len(self.feature_names)]

    @property
    def grouper_features(self):
        if self.grouper is not None:
            return self.grouper.features
        else:
            return []

    @property
    def n_train(self) -> int:
        """Fetch the number of training observations."""
        return len(self.splits["train"])

    @property
    def n_domains(self) -> int:
        """Number of domains, across all sensitive attributes and splits."""
        if self.domain_label_colname is None:
            return 0
        else:
            return self._df[self.domain_label_colname].nunique()

    @property
    def cat_idxs(self) -> List[int]:
        return [i for i, col in enumerate(self._df.columns) if is_categorical(self._df[col])]

    def get_domains(self, split) -> Union[List[str], None]:
        """Fetch a list of the domains."""
        if self.is_domain_split and self._is_valid_split(split):
            split_df = self._get_split_df(split)
            return split_df[self.domain_label_colname].unique()
        else:
            return None

    def _check_data(self):
        """Helper function to check data after all preprocessing/splitting."""
        target = self._post_transform_target_name()
        if not pd.api.types.is_numeric_dtype(self._df[target]):
            logging.warning(
                f"y is of type {self._df[target].dtype}; "
                f"non-numeric types are not accepted by all estimators ("
                f"e.g. xgb.XGBClassifier")
        if self.domain_label_colname:
            assert self.domain_label_colname not in self._df[
                self.feature_names].columns

        if self.grouper and self.grouper.drop:
            for c in self.grouper_features: assert c not in self._df[
                self.feature_names].columns
        return

    @staticmethod
    def _check_data_source(df: pd.DataFrame):
        """Check the data returned by DataSource.get_data()."""
        df = convert_64bit_numeric_cols(df)
        df.reset_index(drop=True, inplace=True)
        return df

    def _initialize_data(self):
        """Load the data/labels/groups from a data source."""
        data = self.data_source.get_data()
        data = self._check_data_source(data)
        data = self.task_config.feature_list.apply_schema(
            data, passthrough_columns=["Split"])
        data = self.preprocessor._dropna(data)
        data = self._apply_grouper(data)
        data = self._generate_splits(data)
        data = self._process_post_split(data)
        self._df = data

        self._init_feature_names(data)
        self._check_data()

        return

    def _apply_grouper(self, data: pd.DataFrame):
        """Apply the grouper, if one is used."""
        if self.grouper is not None:
            return self.grouper.transform(data)
        else:
            return data

    def _post_transform_target_name(self) -> str:
        """Return the 'true', possibly mapped, name of the target feature.

        This is the name that should be used once the features have been
        transformed."""
        target = self.task_config.feature_list.target
        if self.preprocessor_config.map_targets and (
                self.task_config.feature_list[
                    target].name_extended is not None):
            target = self.task_config.feature_list[target].name_extended
        return target

    def _pre_transform_target_name(self) -> str:
        return self.task_config.feature_list.target

    def _init_feature_names(self, data):
        """Set the (data, labels, groups, domain_labels) feature names."""
        target = self._post_transform_target_name()

        data_features = set([x for x in data.columns
                             if x not in self.grouper_features
                             and x != target])
        if self.grouper and (not self.grouper.drop):
            # Retain the group variables as features.
            for x in self.grouper_features: data_features.add(x)

        if isinstance(self.splitter, DomainSplitter):
            domain_split_varname = self.splitter.domain_split_varname

            if self.splitter.drop_domain_split_col and \
                    (domain_split_varname in data_features):
                # Retain the domain split variable as feature in X.
                data_features.remove(domain_split_varname)
        else:
            # Case: domain split is not used; no domain labels exist.
            domain_split_varname = None

        self.feature_names = list(data_features)
        self.target = target
        self.group_feature_names = self.grouper_features
        self.domain_label_colname = domain_split_varname

        return

    def _generate_splits(self, data):
        """Call the splitter to generate splits for the dataset."""
        assert self.splits is None, "attempted to overwrite existing splits."
        self._init_feature_names(data)
        self.splits = self.splitter(
            data=data[self.feature_names],
            labels=data[self._pre_transform_target_name()],
            groups=data[self.group_feature_names],
            domain_labels=data[self.domain_label_colname] \
                if self.domain_label_colname else None)
        if "Split" in data.columns:
            data.drop(columns=["Split"], inplace=True)
        return data

    def _process_post_split(self, data) -> pd.DataFrame:
        """Dataset-specific postprocessing function.

        Conducts any processing required **after** splitting (e.g.
        normalization, drop features needed only for splitting)."""
        passthrough_columns = self.grouper_features

        data = self.preprocessor.fit_transform(
            data,
            self.splits["train"],
            domain_label_colname=self.domain_label_colname,
            target_colname=self.target,
            passthrough_columns=passthrough_columns)

        # If necessary, cast targets to the default type.
        target_name = self._post_transform_target_name()
        if self.preprocessor_config.cast_targets_to_default_type:
            data[target_name] = data[target_name].astype(
                self.preprocessor_config.default_targets_dtype)
        return data

    def _is_valid_split(self, split) -> bool:
        return split in self.splits.keys()

    def _get_split_idxs(self, split):
        self._check_split(split)
        idxs = self.splits[split]
        return idxs

    def _get_split_df(self, split, domain=None) -> pd.DataFrame:

        split_idxs = self._get_split_idxs(split)
        split_df = self._df.iloc[split_idxs]

        if domain is None:
            return split_df

        else:
            assert self.domain_label_colname, 'domain name is required.'
            domain_split_idxs = split_df[self.domain_label_colname] == domain
            return split_df[domain_split_idxs]

    def get_domain_dataloaders(
            self, split, batch_size=2048,
            shuffle=True, infinite=True) -> Dict[Any, DataLoader]:
        """Fetch a dict of {domain_id:DataLoader}."""
        loaders = {}
        split_data = self._get_split_xygd(split)
        assert self.n_domains, "sanity check for a domain-split dataset"

        logging.debug("Domain value counts:\n{}".format(
            split_data[3].value_counts()))

        for domain in sorted(split_data[3].unique()):
            # Boolean vector where True indicates observations in the domain.
            idxs = split_data[3] == domain
            assert idxs.sum() >= batch_size, \
                "sanity check at least one full batch per domain."

            split_domain_data = [df[idxs] for df in split_data]
            split_loader = _make_dataloader_from_dataframes(
                split_domain_data, batch_size, shuffle, infinite=infinite)
            loaders[domain] = split_loader
        return loaders

    def get_dataset_baseline_metrics(self, split):

        X_tr, y_tr, g, _ = self.get_pandas(split)
        n_by_y = pd.value_counts(y_tr).to_dict()
        y_maj = pd.value_counts(y_tr).idxmax()
        # maps {class_label: p_class_label}
        p_y = pd.value_counts(y_tr, normalize=True).to_dict()

        p_y_by_sens = pd.crosstab(y_tr, [X_tr[c] for c in g],
                                  normalize='columns').to_dict()
        n_y_by_sens = pd.crosstab(y_tr, [X_tr[c] for c in g]).to_dict()
        n_by_sens = pd.crosstab(g.iloc[:, 0], g.iloc[:, 1]).unstack().to_dict()
        return {"y_maj": y_maj,
                "n_by_y": n_by_y,
                "p_y": p_y,
                "p_y_by_sens": p_y_by_sens,
                "n_y_by_sens": n_y_by_sens,
                "n_by_sens": n_by_sens}

    def subgroup_majority_classifier_performance(self, split):
        """Compute overall and worst-group acc of a subgroup-conditional
        majority-class classifier."""
        baseline_metrics = self.get_dataset_baseline_metrics(split)
        sensitive_subgroup_accuracies = []
        sensitive_subgroup_n_correct = []
        for sens, n_y_by_sens in baseline_metrics["n_y_by_sens"].items():
            p_y_by_sens = baseline_metrics["p_y_by_sens"][sens]
            y_max = 1 if n_y_by_sens[1] > n_y_by_sens[0] else 0
            p_y_max = p_y_by_sens[y_max]
            n_y_max = n_y_by_sens[y_max]
            sensitive_subgroup_n_correct.append(n_y_max)
            sensitive_subgroup_accuracies.append(p_y_max)
        n = sum(baseline_metrics["n_by_y"].values())
        overall_acc = np.sum(sensitive_subgroup_n_correct) / n
        return overall_acc, min(sensitive_subgroup_accuracies)

    def evaluate_predictions(self, preds, split):
        _, labels, groups, _ = self.get_pandas(split)
        metrics = metrics_by_group(labels, preds, groups, suffix=split)
        # Add baseline metrics.
        metrics["majority_baseline_" + split] = max(labels.mean(),
                                                    1 - labels.mean())
        sm_overall_acc, sm_wg_acc = self.subgroup_majority_classifier_performance(
            split)
        metrics["subgroup_majority_overall_acc_" + split] = sm_overall_acc
        metrics["subgroup_majority_worstgroup_acc_" + split] = sm_wg_acc
        return metrics

    def is_cached(self) -> bool:
        base_dir = os.path.join(self.config.cache_dir, self.uid)
        if os.path.exists(os.path.join(base_dir, "info.json")):
            return True
        else:
            return False

    @property
    def base_dir(self) -> str:
        return os.path.join(self.config.cache_dir, self.uid)

    def _get_info(self) -> Dict[str, Any]:
        return {
            'target': self.target,
            'domain_label_colname': self.domain_label_colname,
            'domain_label_values': self._df[
                self.domain_label_colname].unique().tolist() \
                if self.domain_label_colname else None,
            'group_feature_names': self.group_feature_names,
            'feature_names': self.feature_names,
            'X_shape': self.X_shape,
            'splits': list(self.splits.keys()),
            **{f'n_{s}': len(self.splits[s]) for s in self.splits},
        }

    def _get_schema(self):
        return self._df.dtypes.to_dict()

    def to_sharded(self, rows_per_shard=4096,
                   domains_to_subdirectories: bool = True,
                   file_type="csv"):

        assert file_type in (
            "csv", "arrow", "parquet"), f"file type {file_type} not supported."

        base_dir = self.base_dir

        def initialize_dir(dirname):
            if not os.path.exists(dirname):
                os.makedirs(dirname)

        for split in self.splits:

            logging.info(f"caching task split {split} to {self.base_dir}")

            df = self._get_split_df(split)

            def write_shards(to_shard: DataFrame, dirname: str):
                num_shards = math.ceil(len(to_shard) / rows_per_shard)
                for i in range(num_shards):
                    fp = os.path.join(dirname, f"{split}_{i:05d}.{file_type}")
                    logging.debug('writing file to %s' % fp)
                    start, end = i * rows_per_shard, (i + 1) * rows_per_shard
                    shard_df = to_shard.iloc[start:end]
                    if file_type == "csv":
                        shard_df.to_csv(fp, index=False)
                    elif file_type == "arrow":
                        shard_df.reset_index(drop=True).to_feather(fp)
                    elif file_type == "parquet":
                        shard_df.to_parquet(fp, index=False)

            if self.domain_label_colname and domains_to_subdirectories:
                # Write to {split}/{domain_value}/{shard_filename.csv}
                for domain in sorted(df[self.domain_label_colname].unique()):
                    df_ = df[df[self.domain_label_colname] == domain]
                    domain_dir = self.get_cache_dir(split, domain)
                    initialize_dir(domain_dir)
                    write_shards(df_, domain_dir)
            elif self.domain_label_colname:
                # Write to {split}/all_{split}_subdomains/{shard_filename.csv}
                shared_domain_dir = self.get_cache_dir(
                    split, f"all_{split}_subdomains")
                initialize_dir(shared_domain_dir)
                write_shards(df, shared_domain_dir)
            else:
                # Write to {split}/{shard_filename.csv}
                outdir = self.get_cache_dir(split)
                initialize_dir(outdir)
                write_shards(df, outdir)

        # write metadata
        schema = self._get_schema()
        with open(os.path.join(base_dir, "schema.pickle"), "wb") as f:
            pickle.dump(schema, f)

        ds_info = self._get_info()
        with open(os.path.join(base_dir, "info.json"), "w") as f:
            f.write(json.dumps(ds_info))


class CachedDataset(Dataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.domain_label_values = None
        self.group_feature_names = None

        # Shape of the full data matrix, including domain label and target
        self.X_shape = None

        self.schema = None

        self._load_info_from_cache()

    @property
    def base_dir(self):
        return os.path.join(self.cache_dir, self.uid)

    @property
    def cat_idxs(self) -> List[int]:
        """Fetch indices of categorical features in the data.

        These are indices into the `X` array, with columns ordered according to
        self.feature_names (which is the default ordering provided by all
        TableShift functions).
        """
        features_and_dtypes = [(x, self.schema[x]) for x in self.feature_names]
        idxs = np.nonzero([x[1] == np.int8 for x in features_and_dtypes])[0]
        return idxs.tolist()

    @property
    def features(self) -> List[str]:
        return list(self.schema.keys())

    def _initialize_data(self):
        raise NotImplementedError

    def is_cached(self) -> bool:
        base_dir = os.path.join(self.cache_dir, self.uid)
        if os.path.exists(os.path.join(base_dir, "info.json")):
            return True
        else:
            return False

    def _init_feature_names(self):
        # Hack: since feature names don't match the ordering in the file on
        # disk, we peek at the file on disk; it is critical that the ordering
        # of the feature names matches the ordering in the data.
        f = self._get_split_files("train")[0]
        df = pd.read_csv(f, nrows=1)
        assert set(df.columns) == set(self.schema.keys())
        feature_names = df.columns.tolist()
        feature_names.remove(self.target)
        if isinstance(self.splitter, DomainSplitter) \
                and self.splitter.drop_domain_split_col \
                and self.domain_label_colname in feature_names:
            feature_names.remove(self.domain_label_colname)
        self.feature_names = feature_names

    def _load_info_from_cache(self):
        """Load the dataset metadata from cache (data is lazily loaded)."""
        logging.info(f"loading from {self.base_dir}")
        with open(os.path.join(self.base_dir, "info.json"), "r") as f:
            ds_info = json.loads(f.read())

        for k, v in ds_info.items():
            if k != "feature_names":
                setattr(self, k, v)

        with open(os.path.join(self.base_dir, "schema.pickle"), "rb") as f:
            self.schema = pickle.load(f)

        self._init_feature_names()

    def get_domains(self, split) -> Union[List[str], None]:
        """Fetch a list of the cached domains."""
        dir = os.path.join(self.base_dir, split)
        if not os.path.exists(dir):
            return None
        else:
            domains = os.listdir(dir)
            return sorted(domains)

    @property
    def n_domains(self) -> int:
        """Number of domains, across all sensitive attributes and splits."""
        if self.domain_label_colname is None:
            return 0
        else:
            domains_per_split = [self.get_domains(s) for s in self.splits]
            domains = list(set(d for ds in domains_per_split for d in ds))
            return len(domains)

    def _get_split_files(self, split: str, domain: Optional[str] = None):
        split_cache_dir = self.get_cache_dir(split, domain)
        if not domain:  # case: match files in any domain subdirectory
            fileglob = os.path.join(split_cache_dir, "**", "*.csv")
            files = glob.glob(fileglob, recursive=True)
        else:
            # case: split_cache_dir contains a domain subdirectory, so this
            # will only match the desired domain.
            fileglob = os.path.join(split_cache_dir, "*.csv")
            files = glob.glob(fileglob)

        assert len(files), f"no files detected for split {split} " \
                           f"matching {fileglob}"
        return files

    def _is_valid_split(self, split) -> bool:
        return split in os.listdir(self.base_dir)

    def _get_split_df(self, split, domain=None) -> pd.DataFrame:
        self._check_split(split)
        files = self._get_split_files(split, domain=domain)
        dfs = []
        for f in files:
            dfs.append(pd.read_csv(f))
        df = pd.concat(dfs)

        return df

    def get_ray(self, split, domain=None, num_partitions_per_file=16):
        files = self._get_split_files(split, domain)
        num_partitions = len(files) * num_partitions_per_file
        return ray.data \
            .read_csv(
            files,
            meta_provider=ray.data.datasource.FastFileMetadataProvider()) \
            .repartition(num_partitions)

ACS_YEARS = [2014, 2015, 2016, 2017, 2018]
BRFSS_YEARS = (2015, 2017, 2019, 2021)
NHANES_YEARS = [1999, 2001, 2003, 2005, 2007, 2009, 2011, 2013, 2015, 2017]


BENCHMARK_CONFIGS = {
    "acsfoodstamps": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="DIVISION",
                                domain_split_ood_values=['06']),
        grouper=Grouper({"RAC1P": [1, ], "SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={"acs_task": "acsfoodstamps"}),

    "acsincome": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="DIVISION",
                                domain_split_ood_values=['01']),
        grouper=Grouper({"RAC1P": [1, ], "SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={"acs_task": "acsincome"}),

    "acspubcov": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="DIS",
                                domain_split_ood_values=['1.0']),
        grouper=Grouper({"RAC1P": [1, ], "SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={"acs_task": "acspubcov", "name": "acspubcov",
                                "years": ACS_YEARS}),

    "acsunemployment": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='SCHL',
                                # No high school diploma vs. GED/diploma or higher.
                                domain_split_ood_values=['01', '02', '03', '04',
                                                         '05', '06', '07', '08',
                                                         '09', '10', '11', '12',
                                                         '13', '14', '15']),
        grouper=Grouper({"RAC1P": [1, ], "SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={"acs_task": "acsunemployment"}),

    # ANES, Split by region; OOD is south: (AL, AR, DE, D.C., FL, GA, KY, LA,
    # MD, MS, NC, OK, SC,TN, TX, VA, WV)
    "anes": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='VCF0112',
                                domain_split_ood_values=['3.0']),
        # male vs. all others; white non-hispanic vs. others
        grouper=Grouper({"VCF0104": ["1", ], "VCF0105a": ["1.0", ]},
                        drop=False),
        preprocessor_config=PreprocessorConfig(numeric_features="kbins",
                                               dropna=None),
        tabular_dataset_kwargs={}),

    "brfss_blood_pressure": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="BMI5CAT",
                                # OOD values: [1 underweight, 2 normal weight], [3 overweight, 4 obese]
                                domain_split_ood_values=['3.0', '4.0']),
        grouper=Grouper({"PRACE1": [1, ], "SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig(passthrough_columns=["IYEAR"]),
        tabular_dataset_kwargs={"name": "brfss_blood_pressure",
                                "task": "blood_pressure",
                                "years": BRFSS_YEARS},
    ),

    # "White nonhispanic" (in-domain) vs. all other race/ethnicity codes (OOD)
    "brfss_diabetes": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="PRACE1",
                                domain_split_ood_values=[2, 3, 4, 5, 6],
                                domain_split_id_values=[1, ]),
        grouper=Grouper({"SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig(passthrough_columns=["IYEAR"]),
        tabular_dataset_kwargs={"name": "brfss_diabetes",
                                "task": "diabetes", "years": BRFSS_YEARS},
    ),

    "diabetes_readmission": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='admission_source_id',
                                domain_split_ood_values=[7, ]),
        # male vs. all others; white non-hispanic vs. others
        grouper=Grouper({"race": ["Caucasian", ], "gender": ["Male", ]},
                        drop=False),
        # Note: using min_frequency=0.01 reduces data
        # dimensionality from ~2400 -> 169 columns.
        # This is due to high cardinality of 'diag_*' features.
        preprocessor_config=PreprocessorConfig(min_frequency=0.01),
        tabular_dataset_kwargs={}),

    "heloc": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='ExternalRiskEstimateLow',
                                domain_split_ood_values=[0]),
        grouper=None,
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={"name": "heloc"},
    ),

    "mimic_extract_los_3": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="insurance",
                                domain_split_ood_values=["Medicare"]),

        grouper=Grouper({"gender": ['M'], }, drop=False),
        preprocessor_config=PreprocessorConfig(
            passthrough_columns=_MIMIC_EXTRACT_PASSTHROUGH_COLUMNS),
        tabular_dataset_kwargs={"task": "los_3",
                                "name": "mimic_extract_los_3"}),

    "mimic_extract_mort_hosp": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="insurance",
                                domain_split_ood_values=["Medicare",
                                                         "Medicaid"]),
        grouper=Grouper({"gender": ['M'], }, drop=False),
        preprocessor_config=PreprocessorConfig(
            passthrough_columns=_MIMIC_EXTRACT_PASSTHROUGH_COLUMNS),
        tabular_dataset_kwargs={"task": "mort_hosp",
                                "name": "mimic_extract_mort_hosp"}),

    "nhanes_cholesterol": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='RIDRETH_merged',
                                domain_split_ood_values=[1, 2, 4, 6, 7],
                                domain_split_id_values=[3],
                                ),
        # Group by male vs. all others
        grouper=Grouper({"RIAGENDR": ["1.0", ]}, drop=False),
        preprocessor_config=PreprocessorConfig(
            passthrough_columns=["nhanes_year"],
            numeric_features="kbins"),
        tabular_dataset_kwargs={"nhanes_task": "cholesterol",
                                "years": NHANES_YEARS}),

    "assistments": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='school_id',
                                domain_split_ood_values=[5040.0,
                                                         11502.0,
                                                         11318.0,
                                                         11976.0,
                                                         12421.0,
                                                         12379.0,
                                                         11791.0,
                                                         8359.0,
                                                         12406.0,
                                                         7594.0]),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            passthrough_columns=["skill_id", "bottom_hint", "first_action"],
        ),
        tabular_dataset_kwargs={},
    ),

    "college_scorecard": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='CCBASIC',
                                domain_split_ood_values=[
                                    'Special Focus Institutions--Other special-focus institutions',
                                    'Special Focus Institutions--Theological seminaries, Bible colleges, and other faith-related institutions',
                                    "Associate's--Private For-profit 4-year Primarily Associate's",
                                    'Baccalaureate Colleges--Diverse Fields',
                                    'Special Focus Institutions--Schools of art, music, and design',
                                    "Associate's--Private Not-for-profit",
                                    "Baccalaureate/Associate's Colleges",
                                    "Master's Colleges and Universities (larger programs)"]
                                ),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            # Several categorical features in college scorecard have > 10k
            # unique values; so we label-encode instead of one-hot encoding.
            categorical_features="label_encode",
            # Some important numeric features are not reported by universities
            # in a way that could be systematic (and we would like these included
            # in the sample, not excluded), so we use kbins
            numeric_features="kbins",
            n_bins=100,
            dropna=None,
        ),
        tabular_dataset_kwargs={},
    ),

    "nhanes_lead": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='INDFMPIRBelowCutoff',
                                domain_split_ood_values=[1.]),
        # Race (non. hispanic white vs. all others; male vs. all others)
        grouper=Grouper({"RIDRETH_merged": [3, ], "RIAGENDR": ["1.0", ]},
                        drop=False),
        preprocessor_config=PreprocessorConfig(
            passthrough_columns=["nhanes_year"],
            numeric_features="kbins"),
        tabular_dataset_kwargs={"nhanes_task": "lead", "years": NHANES_YEARS}),

    # LOS >= 47 is roughly the 80th %ile of data.
    "physionet": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='ICULOS',
                                domain_split_gt_thresh=47.0),
        grouper=None,
        preprocessor_config=PreprocessorConfig(numeric_features="kbins",
                                               dropna=None),
        tabular_dataset_kwargs={"name": "physionet"}),
}

EXPERIMENT_CONFIGS = {
    **BENCHMARK_CONFIGS,
    **NON_BENCHMARK_CONFIGS
}

@dataclass
class DatasetConfig:
    cache_dir: str = "tableshift_cache"
    download: bool = True
    random_seed: int = 948324

def get_dataset(name: str, cache_dir: str = "tmp",
                preprocessor_config: Optional[
                    PreprocessorConfig] = None,
                initialize_data: bool = True,
                use_cached: bool = False,
                **kwargs) -> Union[TabularDataset, CachedDataset]:
    """Helper function to fetch a dataset.

    Args:
        name: the dataset name.
        cache_dir: the cache directory to use. TableShift will check for cached
            data files here before downloading.
        preprocessor_config: optional Preprocessor to override the default
            preprocessor config. If using the TableShift benchmark, it is
            recommended to leave this as None to use the default preprocessor.
        initialize_data: passed to TabularDataset constructor.
        use_cached: whether to used cached dataset.
        kwargs: optional kwargs to be passed to TabularDataset; these will
            override their respective kwargs in the experiment config.
        """
    assert name in EXPERIMENT_CONFIGS.keys(), \
        f"Dataset name {name} is not available; choices are: " \
        f"{sorted(EXPERIMENT_CONFIGS.keys())}"

    expt_config = EXPERIMENT_CONFIGS[name]
    dataset_config = DatasetConfig(cache_dir=cache_dir)
    tabular_dataset_kwargs = copy.copy(expt_config.tabular_dataset_kwargs)
    if "name" not in tabular_dataset_kwargs:
        tabular_dataset_kwargs["name"] = name

    if preprocessor_config is None:
        preprocessor_config = expt_config.preprocessor_config

    if not use_cached:
        dset = TabularDataset(
            config=dataset_config,
            splitter=expt_config.splitter,
            grouper=kwargs.get("grouper", expt_config.grouper),
            preprocessor_config=preprocessor_config,
            initialize_data=initialize_data,
            **tabular_dataset_kwargs)
    else:
        dset = CachedDataset(config=dataset_config,
                             splitter=expt_config.splitter,
                             grouper=kwargs.get("grouper", expt_config.grouper),
                             preprocessor_config=preprocessor_config,
                             initialize_data=initialize_data,
                             name=name)
    return dset


# from tableshift.models.training import train


class SklearnStylePytorchModel(ABC, nn.Module):
    """A pytorch model with an sklearn-style interface."""

    def __init__(self):
        super().__init__()

        # Indicator for domain generalization model
        self.domain_generalization = False

        # Indicator for domain adaptation model
        self.domain_adaptation = False

    def _init_optimizer(self):
        """(re)initialize the optimizer."""
        opt_config = {k: self.config[k] for k in OPTIMIZER_ARGS}
        logging.debug(f"initializing optimizer with params {opt_config}")
        self.optimizer = get_optimizer(self, config=opt_config)

    def predict(self, X) -> np.ndarray:
        """sklearn-compatible prediction function."""
        return self(X).detach().cpu().numpy()

    @abstractmethod
    def predict_proba(self, X) -> np.ndarray:
        """sklearn-compatible probability prediction function."""
        raise

    def evaluate(self, eval_loaders: Dict[str, DataLoader], device, exp: Optional[str] = None, tta: bool = False):
        if tta:
             return {str(split): evaluate_tta(self, loader, device, split, exp)
                for split, loader in eval_loaders.items()}
        else:
            return {str(split): evaluate(self, loader, device, split, exp)
                for split, loader in eval_loaders.items()}

    @abstractmethod
    def train_epoch(self,
                    train_loaders: Dict[Any, DataLoader],
                    loss_fn: Callable,
                    device: str,
                    uda_loader: Optional[DataLoader] = None,
                    eval_loaders: Optional[Mapping[str, DataLoader]] = None,
                    # Terminate after this many steps if reached before end
                    # of epoch.
                    max_examples_per_epoch: Optional[int] = None
                    ) -> float:
        """Conduct one epoch of training and return the loss."""
        raise

    def save_checkpoint(self,epoch) -> Checkpoint:
        # Here we save a checkpoint. It is automatically registered with
        # Ray Tune and can be accessed through `session.get_checkpoint()`
        # API in future iterations.
        os.makedirs("model", exist_ok=True)
        torch.save(
            (self.state_dict(), self.optimizer.state_dict()),
            f"model/checkpoint_{epoch}.pt")
        checkpoint = Checkpoint.from_directory("model")
        return checkpoint

    def fit(self, train_loaders: Dict[Any, DataLoader],
            loss_fn,
            device: str,
            n_epochs=1,
            eval_loaders: Optional[Dict[str, DataLoader]] = None,
            tune_report_split: Optional[str] = None,
            max_examples_per_epoch: Optional[int] = None,
            exp: Optional[str] = None) -> dict:
        fit_metrics = defaultdict(list)

        if tune_report_split:
            assert tune_report_split in list(eval_loaders.keys()) + ["train"]

        # TTA stage begin
        if n_epochs == 0:
            tta = True
            logging.info("Start TAT adaptation")
            metrics = self.evaluate(eval_loaders, device=device, exp=exp, tta=tta)
            log_str = f'Epoch {0:03d} ' + ' | '.join(
                f"{k} score: {v:.4f}" for k, v in metrics.items())
            logging.info(log_str)
            fit_metrics = append_by_key(from_dict=metrics, to_dict=fit_metrics)
            return fit_metrics
        # TTA stage ends

        max_ood_score = 0.0
        for epoch in range(1, n_epochs + 1):
            self.train_epoch(train_loaders=train_loaders,
                             loss_fn=loss_fn,
                             eval_loaders=eval_loaders,
                             device=device,
                             max_examples_per_epoch=max_examples_per_epoch)
            metrics = self.evaluate(eval_loaders, device=device)
            log_str = f'Epoch {epoch:03d} ' + ' | '.join(
                f"{k} score: {v:.4f}" for k, v in metrics.items())
            max_ood_score = max(max_ood_score, metrics['ood_test'])

            logging.info(log_str + f'| max_ood_score: {max_ood_score:.5f}')



            checkpoint = self.save_checkpoint(epoch)

            if tune_report_split:
                session.report({"metric": metrics[tune_report_split]},
                               checkpoint=checkpoint)

            fit_metrics = append_by_key(from_dict=metrics, to_dict=fit_metrics)

        return fit_metrics


def get_train_loaders(
        dset: TabularDataset,
        batch_size: int,
        model_name: Optional[str] = None,
        estimator: Optional[SklearnStylePytorchModel] = None,
) -> Dict[Any, torch.utils.data.DataLoader]:
    assert (model_name or estimator) and not (model_name and estimator), \
        "provide either model_name or estimator, but not both."
    if estimator.domain_generalization or is_domain_generalization_model_name(
            model_name):
        train_loaders = dset.get_domain_dataloaders("train", batch_size)
    elif estimator.domain_adaptation or is_domain_adaptation_model_name(
            model_name):
        raise NotImplementedError
    else:
        train_loaders = {"train": dset.get_dataloader("train", batch_size)}
    return train_loaders

def get_eval_loaders(
        dset: TabularDataset,
        batch_size: int,
        model_name: Optional[str] = None,
        estimator: Optional[SklearnStylePytorchModel] = None,
) -> Dict[Any, torch.utils.data.DataLoader]:
    assert (model_name or estimator) and not (model_name and estimator), \
        "provide either model_name or estimator, but not both."
    eval_loaders = {s: dset.get_dataloader(s, 512) for s in
                    dset.eval_split_names}
    if estimator.domain_generalization or is_domain_generalization_model_name(
            model_name):
        train_eval_loaders = dset.get_domain_dataloaders("train", batch_size,
                                                         infinite=False)
        eval_loaders.update(train_eval_loaders)
    elif estimator.domain_adaptation or is_domain_adaptation_model_name(
            model_name):
        raise NotImplementedError
    else:
        eval_loaders["train"] = dset.get_dataloader("train", batch_size)
    return eval_loaders

def _train_pytorch(estimator: SklearnStylePytorchModel, dset: TabularDataset,
                   config=PYTORCH_DEFAULTS,
                   device: str=None,
                   tune_report_split: str = None):
    """Helper function to train a pytorch estimator."""
    if not device:
        device = f"cuda:{torch.cuda.current_device()}" \
             if torch.cuda.is_available() else "cpu"
    logging.debug(f"config is {config}")
    logging.debug(f"estimator is of type {type(estimator)}")
    logging.debug(f"dset name is {dset.name}")
    logging.debug(f"device is {device}")
    logging.debug(f"tune_report_split is {tune_report_split}")

    batch_size = config["batch_size"]
    train_loaders = get_train_loaders(estimator=estimator,
                                      dset=dset, batch_size=batch_size)
    eval_loaders = get_eval_loaders(estimator=estimator,
                                    dset=dset, batch_size=batch_size)

    loss_fn = config["criterion"]

    estimator.to(device)

    estimator.fit(train_loaders, loss_fn,
                  n_epochs=config["n_epochs"],
                  device=device,
                  eval_loaders=eval_loaders,
                  tune_report_split=tune_report_split,
                  max_examples_per_epoch=dset.n_train,
                  exp=config['exp'])
    return estimator

def _train_sklearn(estimator, dset: TabularDataset,
                   tune_report_split: str = None):
    """Helper function to train a sklearn-type estimator."""
    X_tr, y_tr, _, d_tr = dset.get_pandas(split="train")
    if isinstance(estimator, ExponentiatedGradient):
        estimator.fit(X_tr, y_tr, d=d_tr)
    elif isinstance(estimator, WeightedCovariateShiftClassifier):
        X_ood_tr, y_ood_tr, _, _ = dset.get_pandas(split="ood_validation")
        estimator.fit(X_tr, y_tr, X_ood_tr)
    else:
        estimator.fit(X_tr, y_tr)
    logging.info("fitting estimator complete.")

    if tune_report_split:
        X_te, _, _, _ = dset.get_pandas(split=tune_report_split)
        y_hat_te = estimator.predict(X_te)
        metrics = dset.evaluate_predictions(y_hat_te, split=tune_report_split)
        session.report({"metric": metrics[f"accuracy_{tune_report_split}"]})
    return estimator

def train(estimator: Any, dset: TabularDataset, tune_report_split: str = None,
          **kwargs):
    logging.info(f"fitting estimator of type {type(estimator)}")
    logging.info(estimator)
    if isinstance(estimator, torch.nn.Module):
        assert isinstance(
            estimator,
            SklearnStylePytorchModel), \
            f"train() can only be called with SklearnStylePytorchModel; got " \
            f"type {type(estimator)} "
        return _train_pytorch(estimator, dset,
                              tune_report_split=tune_report_split, **kwargs)
    else:
        return _train_sklearn(estimator, dset,
                              tune_report_split=tune_report_split)
    



# from tableshift.models.utils import get_estimator
def get_estimator(model:str, d_out=1, **kwargs):
    """
    Fetch an estimator for training.

    Args:
        model: the string name of the model to use.
        d_out: output dimension of the model (set to 1 for binary classification).
        kwargs: named arguments to pass to the model's class constructor. These
            vary by model; for more details see below. Note that only a specific
            subset of the kwargs will be used; passing arbitrary kwargs not accepted by
            the model's class constructor will result in those kwargs being ignored.
    Returns:
        An instance of the class specified by the `model` string, with
            any hyperparameters set according to kwargs.
    """
    if model == "aldro":
        assert d_out == 1, "assume binary classification."
        return AdversarialLabelDROModel(
            d_in=kwargs["d_in"],
            d_layers=[kwargs["d_hidden"]] * kwargs["num_layers"],
            d_out=d_out,
            dropouts=kwargs["dropouts"],
            activation=kwargs["activation"],
            n_groups=2,
            **{k: kwargs[k] for k in OPTIMIZER_ARGS},
            eta_pi=kwargs["eta_pi"],
            r=kwargs["r"],
        )

    elif model == "dann":
        return DANNModel(d_in=kwargs["d_in"],
                         d_layers=[kwargs["d_hidden"]] * kwargs[
                             "num_layers"],
                         d_out=d_out,
                         dropouts=kwargs["dropouts"],
                         activation=kwargs["activation"],
                         lr_d=kwargs["lr_d"],
                         weight_decay_d=kwargs["weight_decay_d"],
                         lr_g=kwargs["lr_g"],
                         weight_decay_g=kwargs["weight_decay_g"],
                         d_steps_per_g_step=kwargs["d_steps_per_g_step"],
                         grad_penalty=kwargs["grad_penalty"],
                         loss_lambda=kwargs["loss_lambda"],
                         )

    elif model == "deepcoral":
        return DeepCoralModel(d_in=kwargs["d_in"],
                              d_layers=[kwargs["d_hidden"]] * kwargs[
                                  "num_layers"],
                              d_out=d_out,
                              dropouts=kwargs["dropouts"],
                              activation=kwargs["activation"],
                              mmd_gamma=kwargs["mmd_gamma"],
                              **{k: kwargs[k] for k in OPTIMIZER_ARGS})
    elif model == "expgrad":
        return ExponentiatedGradient(**kwargs)

    elif model == "ft_transformer":
        tconfig = FTTransformerModel.get_default_transformer_config()

        tconfig["last_layer_query_idx"] = [-1]
        tconfig["d_out"] = 1
        params_to_override = ("n_blocks", "residual_dropout", "d_token",
                              "attention_dropout", "ffn_dropout")
        for k in params_to_override:
            tconfig[k] = kwargs[k]

        tconfig["ffn_d_hidden"] = int(kwargs["d_token"] * kwargs["ffn_factor"])

        # Fixed as in https://arxiv.org/pdf/2106.11959.pdf
        tconfig['attention_n_heads'] = 8

        # Hacky way to construct a FTTransformer model
        model = FTTransformerModel._make(
            n_num_features=kwargs["n_num_features"],
            cat_cardinalities=kwargs["cat_cardinalities"],
            transformer_config=tconfig)
        tconfig.update({k: kwargs[k] for k in OPTIMIZER_ARGS})
        model.config = copy.deepcopy(tconfig)
        model._init_optimizer()

        return model

    elif model == "group_dro":
        return DomainGroupDROModel(
            d_in=kwargs["d_in"],
            d_layers=[kwargs["d_hidden"]] * kwargs["num_layers"],
            d_out=d_out,
            dropouts=kwargs["dropouts"],
            activation=kwargs["activation"],
            group_weights_step_size=kwargs["group_weights_step_size"],
            n_groups=kwargs["n_groups"],
            **{k: kwargs[k] for k in OPTIMIZER_ARGS},
        )

    elif model == "histgbm":
        return HistGradientBoostingClassifier(**kwargs)

    elif model == "irm":
        return IRMModel(
            d_in=kwargs["d_in"],
            d_layers=[kwargs["d_hidden"]] * kwargs["num_layers"],
            d_out=d_out,
            dropouts=kwargs["dropouts"],
            activation=kwargs["activation"],
            irm_lambda=kwargs['irm_lambda'],
            irm_penalty_anneal_iters=kwargs['irm_penalty_anneal_iters'],
            **{k: kwargs[k] for k in OPTIMIZER_ARGS}, )

    elif model == "label_group_dro":
        return LabelGroupDROModel(
            d_in=kwargs["d_in"],
            d_layers=[kwargs["d_hidden"]] * kwargs["num_layers"],
            d_out=d_out,
            dropouts=kwargs["dropouts"],
            activation=kwargs["activation"],
            group_weights_step_size=kwargs["group_weights_step_size"],
            n_groups=kwargs["n_groups"],
            **{k: kwargs[k] for k in OPTIMIZER_ARGS},
        )
    elif model == "lightgbm":
        return LGBMClassifier(**kwargs)

    elif model == "mixup":
        return MixUpModel(
            d_in=kwargs["d_in"],
            d_layers=[kwargs["d_hidden"]] * kwargs["num_layers"],
            d_out=d_out,
            dropouts=kwargs["dropouts"],
            activation=kwargs["activation"],
            mixup_alpha=kwargs["mixup_alpha"],
            **{k: kwargs[k] for k in OPTIMIZER_ARGS}
        )

    elif model == "mlp" or model == "dro":
        return MLPModel(d_in=kwargs["d_in"],
                        d_layers=[kwargs["d_hidden"]] * kwargs["num_layers"],
                        d_out=d_out,
                        dropouts=kwargs["dropouts"],
                        activation=kwargs["activation"],
                        **{k: kwargs[k] for k in OPTIMIZER_ARGS})

    elif model == "mmd":
        return MMDModel(d_in=kwargs["d_in"],
                        d_layers=[kwargs["d_hidden"]] * kwargs[
                            "num_layers"],
                        d_out=d_out,
                        dropouts=kwargs["dropouts"],
                        activation=kwargs["activation"],
                        mmd_gamma=kwargs["mmd_gamma"],
                        **{k: kwargs[k] for k in OPTIMIZER_ARGS})

    elif model == "node":
        return NodeModel(d_in=kwargs["d_in"],
                         tree_dim=kwargs["tree_dim"],
                         depth=kwargs["depth"],
                         num_layers=kwargs["num_layers"],
                         total_tree_count=kwargs["total_tree_count"],
                         **{k: kwargs[k] for k in OPTIMIZER_ARGS})

    elif model == "resnet":
        d_hidden = kwargs["d_main"] * kwargs["hidden_factor"]
        return ResNetModel(
            d_in=kwargs["d_in"],
            n_blocks=kwargs["n_blocks"],
            d_main=kwargs["d_main"],
            d_hidden=d_hidden,
            dropout_first=kwargs["dropout_first"],
            dropout_second=kwargs["dropout_second"],
            normalization='BatchNorm1d',
            activation=kwargs["activation"],
            d_out=d_out,
            **{k: kwargs[k] for k in OPTIMIZER_ARGS},
        )

    elif model == "saint":
        # Apply the same hparam-setting logic as in SAINT code; see
        # https://github.com/somepago/saint/blob/e288e84c77a54cfd2ffb55a53678fb7cbbb16630/train.py#L91

        if kwargs["d_in"] > 100:
            dim = min(8, kwargs["dim"])
            logging.info(f"setting SAINT embedding dim={dim} for data with "
                         f"d_in>100.")
        else:
            dim = kwargs["dim"]

        if kwargs["attentiontype"] != "col":
            depth = 1
            heads = min(4, kwargs["heads"])
            attn_dropout = 0.8
            logging.info(
                f"setting fixed SAINT params"
                f"for attentiontype {kwargs['attentiontype']}:"
                f"depth={depth}, heads={heads}, attn_dropout={attn_dropout}")
        else:
            depth = kwargs["depth"]
            heads = kwargs["heads"]
            attn_dropout = 0.

        # This is not in SAINT, but is necessary since a few TableShift datasets
        # have much larger dimensionality than those in SAINT paper.
        if kwargs['d_in'] > 1000:
            mlp_hidden_mults = (2, 1)
        else:
            mlp_hidden_mults = (4, 2)

        return SaintModel(
            categories=kwargs["categories"],
            cat_idxs=kwargs["cat_idxs"],
            num_continuous=kwargs["d_in"] - len(kwargs["cat_idxs"]),
            dim=dim,
            depth=depth,
            heads=heads,
            attentiontype=kwargs["attentiontype"],
            attn_dropout=attn_dropout,
            y_dim=1,
            mlp_hidden_mults=mlp_hidden_mults,
            **{k: kwargs[k] for k in OPTIMIZER_ARGS})

    elif model == "tabtransformer":
        return TabTransformerModel(
            categories=kwargs["categories"],
            cat_idxs=kwargs["cat_idxs"],
            num_continuous=kwargs["d_in"] - len(kwargs["cat_idxs"]),
            dim=kwargs["dim"],
            dim_out=1,
            depth=kwargs["depth"],
            heads=kwargs["heads"],
            attn_dropout=kwargs["attn_dropout"],
            ff_dropout=kwargs["ff_dropout"],
            mlp_hidden_mults=kwargs["mlp_hidden_mults"],
            **{k: kwargs[k] for k in OPTIMIZER_ARGS})

    elif model == "vrex":
        return VRExModel(
            d_in=kwargs["d_in"],
            d_layers=[kwargs["d_hidden"]] * kwargs["num_layers"],
            d_out=d_out,
            dropouts=kwargs["dropouts"],
            activation=kwargs["activation"],
            vrex_penalty_anneal_iters=kwargs["vrex_penalty_anneal_iters"],
            vrex_lambda=kwargs["vrex_lambda"],
            **{k: kwargs[k] for k in OPTIMIZER_ARGS})

    elif model == "wcs":
        # Weighted Covariate Shift classifier.
        return WeightedCovariateShiftClassifier(**kwargs)

    elif model == "xgb":
        return xgb.XGBClassifier(**kwargs)

    else:
        raise NotImplementedError(f"model {model} not implemented.")


DOMAIN_GENERALIZATION_MODEL_NAMES = ["dann", "deepcoral", "irm", "mixup", "mmd",
                                     "vrex"]
DOMAIN_ADAPTATION_MODEL_NAMES = []
DOMAIN_ROBUSTNESS_MODEL_NAMES = ["group_dro", "dro"]
LABEL_ROBUSTNESS_MODEL_NAMES = ["aldro", "label_group_dro"]
SKLEARN_MODEL_NAMES = ("expgrad", "histgbm", "lightgbm", "wcs", "xgb")
BASELINE_MODEL_NAMES = ["ft_transformer", "mlp", "resnet", "node", "saint",
                        "tabtransformer"]
PYTORCH_MODEL_NAMES = BASELINE_MODEL_NAMES \
                      + DOMAIN_ROBUSTNESS_MODEL_NAMES \
                      + DOMAIN_GENERALIZATION_MODEL_NAMES \
                      + DOMAIN_ADAPTATION_MODEL_NAMES \
                      + LABEL_ROBUSTNESS_MODEL_NAMES

def is_pytorch_model_name(model: str) -> bool:
    """Helper function to determine whether a model name is a pytorch model.

    See description of is_pytorch_model() above."""
    if model=="catboost":
        logging.warning("Catboost models are not suported in Ray hyperparameter training."
                        " Instead, use the provided catboost-specific script.")
    is_sklearn = model in SKLEARN_MODEL_NAMES
    is_pt = model in PYTORCH_MODEL_NAMES
    assert is_sklearn or is_pt, f"unknown model name {model}"
    return is_pt


# from tableshift.models.default_hparams import get_default_config
def get_default_config(model: str, dset: TabularDataset) -> dict:
    """Get a default config for a model, by name."""
    config = _DEFAULT_CONFIGS.get(model, {})
    model_is_pt = is_pytorch_model_name(model)

    d_in = dset.X_shape[1]
    if model_is_pt and model != "ft_transformer":
        config.update({"d_in": d_in,
                       "activation": "ReLU"})
    elif model_is_pt:
        config.update({"n_num_features": d_in})

    if model in ("tabtransformer", "saint"):
        cat_idxs = dset.cat_idxs
        config["cat_idxs"] = cat_idxs
        config["categories"] = [2] * len(cat_idxs)

    # Set the training objective and any associated hypperparameters.
    if model == "dro":
        config["criterion"] = DROLoss(size=config["size"],
                                      reg=config["reg"],
                                      geometry=config["geometry"],
                                      max_iter=config["max_iter"])
    elif model == "group_dro":
        config["n_groups"] = dset.n_domains
        config["criterion"] = GroupDROLoss(n_groups=dset.n_domains)

    elif model == "label_group_dro":
        config["n_groups"] = 2  # assume binary classification
        config["criterion"] = GroupDROLoss(n_groups=2)


    elif model_is_pt:
        config["criterion"] = F.binary_cross_entropy_with_logits

    if model_is_pt and model != "dann":
        # Note: for DANN model, lr and weight decay are set separately for D
        # and G.
        config.update({"lr": 0.01,
                       "weight_decay": 0.01,
                       })

    # Do not overwrite batch size or epochs if they are set in the default
    # config for the model.
    if "batch_size" not in config and model_is_pt:
        config["batch_size"] = DEFAULT_BATCH_SIZE
    if "n_epochs" not in config and model_is_pt:
        config["n_epochs"] = 0

    if model == "saint" and d_in > 100:
        # same batch size setting logic as in SAINT code:
        # https://github.com/somepago/saint/blob/e288e84c77a54cfd2ffb55a53678fb7cbbb16630/train.py#LL95C5-L95C43
        config["batch_size"] = min(64, config["batch_size"])
    return config
##### IMPORTS ###########

LOG_LEVEL = logging.DEBUG

logger = logging.getLogger()
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    level=LOG_LEVEL,
    datefmt='%Y-%m-%d %H:%M:%S')


def main(experiment, cache_dir, model, debug: bool):
    if debug:
        print("[INFO] running in debug mode.")
        experiment = "_debug"

    dset = get_dataset(experiment, cache_dir)
    X, y, _, _ = dset.get_pandas("train")
    config = get_default_config(model, dset)
    config['exp'] = experiment
    estimator = get_estimator(model, **config)
    path = './models/'
    if experiment == 'anes':
        path = path+'anes/'
    elif experiment == 'assistments':
        path = path+'assistments/'
    elif experiment =='heloc':
        path = path+'heloc/'
    elif experiment == 'diabetes_readmission':
        path = path+'diabetes'
    else:
        print("please check experiment name!")
        raise ValueError("please check experiment name!")
    
    if model == 'mlp':
        path = path+'/mlp/checkpoint.pt'
    elif model =='tabtransformer':
        path = path + '/tabtrans/checkpoint.pt'
    elif model =='ft_transformer':
        path = path + '/fttrans/checkpoint.pt'
    para,_ = torch.load(path)
    print(estimator.load_state_dict(para))
    estimator = train(estimator, dset, config=config)
    print(type(estimator))

    if not isinstance(estimator, torch.nn.Module):
        # Case: non-pytorch estimator; perform test-split evaluation.
        test_split = "ood_test" if dset.is_domain_split else "test"
        # Fetch predictions and labels for a sklearn model.
        X_te, y_te, _, _ = dset.get_pandas(test_split)
        yhat_te = estimator.predict(X_te)
        acc = accuracy_score(y_true=y_te, y_pred=yhat_te)
        print(f"training completed! {test_split} accuracy: {acc:.4f}")

    else:
        # Case: pytorch estimator; eval is already performed + printed by train().
        print("training completed!")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="tmp",
                        help="Directory to cache raw data files to.")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Whether to run in debug mode. If True, various "
                             "truncations/simplifications are performed to "
                             "speed up experiment.")
    parser.add_argument("--experiment", default="diabetes_readmission",
                        help="Experiment to run. Overridden when debug=True.")
    parser.add_argument("--model", default="mlp",
                        help="model to use.")
    args = parser.parse_args()
    main(**vars(args))


