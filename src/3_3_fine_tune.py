import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import logging
from pathlib import Path
from sklearn.impute import SimpleImputer
from src.utils import get_setting
from functools import reduce
from math import ceil
import os
from shutil import rmtree
from typing import Tuple


def prepare_features() -> Tuple[str, str]:
    current_feature_list = settings["args"].cfl
    if len(current_feature_list) == 0:
        current_feature_list = []
        cfl = ""
    else:
        current_feature_list = current_feature_list.split(",")
        cfl = f" -cfl " + ",".join(current_feature_list[:-1])

    feature_path = f'../{settings["args"].in_name}/{settings["args"].feature_path}'
    features = pd.read_csv(feature_path)
    features = features[["id"] + current_feature_list]

    feature_path = f'../{settings["args"].in_name}/temp_f_bin.csv'
    features.to_csv(feature_path, index=False)
    del features

    return cfl, feature_path


def lgb_fine_tune() -> None:

    name = settings["args"].name
    in_name, out_name = settings["args"].in_name, settings["args"].out_name
    analytics_path = settings["curr_dir"].parent / Path("analytics")

    pos_max_bin = [255]  # [400, 700, 1600]
    pos_lrate = [0.0175]
    pos_num_leaves = [31]  # [40, 80, 160]
    pos_min_data_in_leaf = [20, 100, 200, 400, 800, 1600, 3200]

    params = np.array(
        np.meshgrid(pos_max_bin, pos_lrate, pos_num_leaves, pos_min_data_in_leaf)
    ).T.reshape(-1, 4)

    for ind, param in enumerate(params):
        max_bin, lrate, num_leaves, min_data_in_leaf = (
            param[0],
            param[1],
            param[2],
            param[3],
        )
        logging.info(
            f"{ind + 1} :: LGB with (max_bin = {max_bin}, lrate = {lrate}, num_leaves = {num_leaves}, min_data_in_leaf = {min_data_in_leaf}) :: processing"
        )

        if name != "":
            get_name = "lgb_ft1_" + name
        else:
            get_name = "lgb_ft1"

        os.system(
            f"python 3_0_create_block_var.py -feature_path {feature_path} \
            -experiment_name {get_name} \
            -type_of_model lgb \
            -in_name {in_name} -out_name {out_name} \
            {cfl} \
            -max_bin {str(max_bin)} \
            -lrate {str(lrate)} \
            -num_leaves {str(num_leaves)} \
            -min_data_in_leaf {str(min_data_in_leaf)} \
            &> file"
        )
        Path("file").unlink()

        block_vars_new = pd.read_csv(analytics_path / f"block_vars_{get_name}.csv")
        block_vars_new["max_bin"] = [max_bin]
        block_vars_new["lrate"] = [lrate]
        block_vars_new["num_leaves"] = [num_leaves]
        block_vars_new["min_data_in_leaf"] = [min_data_in_leaf]
        (analytics_path / f"block_vars_{get_name}.csv").unlink()

        try:
            block_vars = (
                pd.concat([block_vars, block_vars_new], axis=0)
                .sort_values(by=["confidence_lower"], ascending=False)
                .reset_index(drop=True)
            )
        except NameError:
            block_vars = block_vars_new

        if name != "":
            get_name = "_lgb_ftm_" + name
        else:
            get_name = "_lgb_ftm"
        block_vars.to_csv(analytics_path / f"block_vars{get_name}.csv", index=False)
    return None


def dnn_fine_tune() -> None:

    name = settings["args"].name
    in_name, out_name = settings["args"].in_name, settings["args"].out_name
    analytics_path = settings["curr_dir"].parent / Path("analytics")

    pos_msl = [500, 1_000, 2_000, 4_000, 8_000, 16_000]  # [10, 50, 100, 250, 500]
    pos_first_units = [256]  # [1024, 512, 256]
    params = np.array(np.meshgrid(pos_msl, pos_first_units)).T.reshape(-1, 2)

    for ind, param in enumerate(params):
        msl, first_units = param[0], param[1]
        logging.info(
            f"{ind + 1} :: DNN with (msl = {msl}, first_units = {first_units}) :: processing"
        )

        if name != "":
            get_name = "dnn_ft1_" + name
        else:
            get_name = "dnn_ft1"

        os.system(
            f"python 3_0_create_block_var.py -feature_path {feature_path} \
            -experiment_name {get_name} \
            -type_of_model dnn \
            -in_name {in_name} -out_name {out_name} \
            {cfl} \
            -msl {str(msl)} \
            -first_units {str(first_units)} -d"  # \
            # &> file"
        )
        # Path("file").unlink()

        block_vars_new = pd.read_csv(analytics_path / f"block_vars_{get_name}.csv")
        block_vars_new["msl"] = [msl]
        block_vars_new["first_units"] = [first_units]
        (analytics_path / f"block_vars_{get_name}.csv").unlink()

        try:
            block_vars = (
                pd.concat([block_vars, block_vars_new], axis=0)
                .sort_values(by=["confidence_lower"], ascending=False)
                .reset_index(drop=True)
            )
        except NameError:
            block_vars = block_vars_new

        if name != "":
            get_name = "_dnn_ftm_" + name
        else:
            get_name = "_dnn_ftm"
        block_vars.to_csv(analytics_path / f"block_vars{get_name}.csv", index=False)
    return None


if __name__ == "__main__":
    settings = get_setting(
        [
            ["-type_of_model", "", ""],
            ["-feature_path", "Path of feature df", ""],
            ["-name", "Name of experiment", ""],
            ["-cfl", "Current feature list", ""],
        ]
    )
    cfl, feature_path = prepare_features()

    if settings["args"].type_of_model == "lgb":
        lgb_fine_tune()
    elif settings["args"].type_of_model == "dnn":
        dnn_fine_tune()

    Path(feature_path).unlink()
