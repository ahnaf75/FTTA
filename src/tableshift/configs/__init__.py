"""Unified experiment configuration registry."""

from dataclasses import dataclass
from typing import Any, Dict, Union

from tableshift.configs.experiment_defaults import (
    DEFAULT_ID_TEST_SIZE,
    DEFAULT_ID_VAL_SIZE,
    DEFAULT_OOD_VAL_SIZE,
    DEFAULT_RANDOM_STATE,
)
from tableshift.core.features import PreprocessorConfig
from tableshift.core.grouper import Grouper
from tableshift.core.splitter import DomainSplitter, FixedSplitter, RandomSplitter, Splitter
from tableshift.datasets.acs import ACS_YEARS
from tableshift.datasets.brfss import BRFSS_YEARS
from tableshift.datasets.mimic_extract import MIMIC_EXTRACT_STATIC_FEATURES
from tableshift.datasets.mimic_extract_feature_lists import MIMIC_EXTRACT_SHARED_FEATURES
from tableshift.datasets.nhanes import NHANES_YEARS


@dataclass
class ExperimentConfig:
    splitter: Splitter
    grouper: Union[Grouper, None]
    preprocessor_config: PreprocessorConfig
    tabular_dataset_kwargs: Dict[str, Any]

# We passthrough all non-static columns because we use
# MIMIC-extract's default preprocessing/imputation and do not
# wish to modify it for these features
# (static features are not preprocessed by MIMIC-extract). See
# tableshift.datasets.mimic_extract.preprocess_mimic_extract().
_MIMIC_EXTRACT_PASSTHROUGH_COLUMNS = [
    f for f in MIMIC_EXTRACT_SHARED_FEATURES.names
    if f not in MIMIC_EXTRACT_STATIC_FEATURES.names]

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

GRINSTAJN_TEST_SIZE = 0.21

GRINSZTAJN_VAL_SIZE = 0.09
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

EXPERIMENT_CONFIGS = {
    **BENCHMARK_CONFIGS,
    **NON_BENCHMARK_CONFIGS,
}

__all__ = [
    "ExperimentConfig",
    "BENCHMARK_CONFIGS",
    "NON_BENCHMARK_CONFIGS",
    "EXPERIMENT_CONFIGS",
]
