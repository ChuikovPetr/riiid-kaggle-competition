import functools
import gc
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Callable

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow_addons as tfa
from IPython.display import clear_output
from keras.regularizers import L1L2
from multiprocess import Pool
from multiprocess.dummy import Pool as ThreadPool
from tqdm import tqdm


from ccf.datasets import get_sample, get_initial_setting, get_left_right_name, get_std
from ccf.models import ModelLSTM, ModelCNN, ModelLSTM_FCN, ModelRF, ModelDNN, ModelLGB
from ccf.metrics import get_gini
from ccf.utils import delete_objs, cuttoff_ts_in_df, natural_reindex, sort_df
from ccf.analytics import (
    greedy_feature_selection,
    search_architecture,
    get_analytics_row,
)
from ccf.ZOO import *
from ccf.callbacks import FrequencyCallback, FrequencyEpoch
from ccf.preprocess import get_sample_2d_lgb, get_sample_2d_bin
from tensorflow.keras.callbacks import EarlyStopping


from src.utils import get_setting


def create_block_var_lgb(
    features_path: Path,
    experiment_name: str,
    current_feature_list: str,
    max_bin: int,
    learning_rate: float,
    num_leaves: int,
    min_data_in_leaf: int,
) -> None:
    if len(current_feature_list) == 0:
        current_feature_list = []
    else:
        current_feature_list = current_feature_list.split(",")

    count_folds = 1

    base_path_train = settings["input_dir"] / Path("X.csv")
    base_path_val_train = settings["input_dir"] / Path("val.csv")
    base_path_val = settings["input_dir"] / Path("val.csv")

    count_obs_train = None
    count_obs_val = None
    count_obs_val_train = None

    features = pd.read_csv(features_path, nrows=1)
    list_candidates = list(
        set(features.columns) - set(current_feature_list) - set(["id"])
    )
    list_candidates.sort()

    selection_rule = {"field_name": "confidence_lower", "ascending": False}

    dict_fields, _, _ = get_initial_setting(features_path, count_cuttoff=0)

    analytics_path = settings["input_dir"] / Path("analytics")

    model_class = lambda train_matrix_shape, name: ModelLGB(
        save_path=settings["curr_dir"].parent / Path("saved_models"),
        name=name,
        metric=get_gini,
        learning_setting={
            "verbose_eval": 50,
            "num_boost_round": 10_000,
            "early_stopping_rounds": 50,
            "params": {
                "num_leaves": 131_072,
                "max_bin": 256,
                "learning_rate": 0.05,
                "objective": "binary",
                "metric": "auc",
                "max_depth": 8,
                "feature_fraction": 0.6,
            },
        },
    )

    get_sample_func = lambda possible_feature_list, base_path, count_obs, scaler: get_sample_2d_lgb(
        possible_feature_list,
        base_path,
        count_obs,
        features_path,
        categoricals=[
            "part",
            "prior_question_had_explanation",
            "kmean_cluster",
            "content_id",
            "lag_part_bool",
            "lag_clu_bool",
            "lag_answ_and_part_bool",
            "lag_answ_and_clu_bool",
            "lag_expl_and_part_bool",
            "lag_expl_and_clu_bool",
            "lag_expl_and_part_bool_not_corr",
            "lag_expl_and_clu_bool_not_corr",
        ],
    )

    greedy_feature_selection(
        current_feature_list,
        list_candidates,
        dict_fields,
        count_folds,
        count_obs_train,
        count_obs_val,
        experiment_name,
        model_class,
        analytics_path,
        selection_rule,
        get_sample_func,
        base_path_train,
        base_path_val,
        base_path_val_train,
        count_obs_val_train,
        print_iteration=False,
        one_iter=True,
    )

    return None


def create_block_var_dnn(
    features_path: Path,
    experiment_name: str,
    current_feature_list: str,
    min_samples_leaf: int,
    first_units: int,
) -> None:
    if len(current_feature_list) == 0:
        current_feature_list = []
    else:
        current_feature_list = current_feature_list.split(",")

    count_folds = 1

    base_path_train = settings["input_dir"] / Path("X.csv")
    base_path_val_train = settings["input_dir"] / Path("train_val.csv")
    base_path_val = settings["input_dir"] / Path("val.csv")

    count_obs_train = None
    count_obs_val = None
    count_obs_val_train = None

    features = pd.read_csv(features_path, nrows=1)
    list_candidates = list(
        set(features.columns) - set(current_feature_list) - set(["id"])
    )
    list_candidates.sort()

    selection_rule = {"field_name": "confidence_lower", "ascending": False}

    dict_fields, _, _ = get_initial_setting(features_path, count_cuttoff=0)

    analytics_path = settings["curr_dir"].parent / Path("analytics")

    learning_setting = {
        "batch_size": 1024,
        "epochs": 100,
        "custom": True,
        "callbacks": lambda validation_data: FrequencyEpoch(
            compare_field="confidence_lower",
            path_stats=analytics_path / Path("pred_epochs_stats.csv"),
            min_epoch=0,
            max_epoch_after=3,
            validation_data=validation_data,
        ),
    }

    model_class = lambda train_matrix_shape, name: ModelDNN(
        input_shape=train_matrix_shape,
        fl_SWA=True,
        save_path=settings["curr_dir"].parent / Path("saved_models"),
        name=name,
        metric=get_gini,
        learning_setting=learning_setting,
        setting={
            "count_dnn_layers": 10,
            "first_units": first_units,
            "min_units": 10,
            "dropout": 0.5,
            "last_dropout": True,
            "kernel_constraint": max_norm(3),
            "l1": 0.0,
            "l2": 0.0,
        },
    )

    get_sample_func = lambda possible_feature_list, base_path, count_obs, scaler: get_sample_2d_bin(
        possible_feature_list,
        base_path,
        count_obs,
        features_path,
        bins_dict=scaler,
        min_samples_leaf=min_samples_leaf,
    )

    greedy_feature_selection(
        current_feature_list,
        list_candidates,
        dict_fields,
        count_folds,
        count_obs_train,
        count_obs_val,
        experiment_name,
        model_class,
        analytics_path,
        selection_rule,
        get_sample_func,
        base_path_train,
        base_path_val,
        base_path_val_train,
        count_obs_val_train,
        print_iteration=False,
        one_iter=True,
    )

    return None


if __name__ == "__main__":
    settings = get_setting(
        [
            ["-feature_path", "Path of feature df", ""],
            ["-experiment_name", "Name of experiment", ""],
            ["-type_of_model", "", "lgb"],
            ["-cfl", "Current feature list", ""],
            ["-max_bin", "Maximum number of bins", "255"],
            ["-lrate", "Learning rate", "0.1"],
            ["-num_leaves", "Number of leaves", "31"],
            ["-min_data_in_leaf", "Minimum observations in leaves", "20"],
            ["-msl", "min_samples_leaf", "3_000"],
            ["-first_units", "Count units in first layer", "256"],
        ]
    )

    if settings["args"].type_of_model == "lgb":
        create_block_var_lgb(
            Path(settings["args"].feature_path),
            settings["args"].experiment_name,
            settings["args"].cfl,
            int(eval(settings["args"].max_bin)),
            eval(settings["args"].lrate),
            int(eval(settings["args"].num_leaves)),
            int(eval(settings["args"].min_data_in_leaf)),
        )

    elif settings["args"].type_of_model == "dnn":
        create_block_var_dnn(
            Path(settings["args"].feature_path),
            settings["args"].experiment_name,
            settings["args"].cfl,
            int(eval(settings["args"].msl)),
            int(eval(settings["args"].first_units)),
        )
