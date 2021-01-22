import pandas as pd
from typing import List, Dict, Tuple
import logging
from pathlib import Path
from sklearn.impute import SimpleImputer
from src.utils import get_setting
from functools import reduce
from math import ceil
import os
from shutil import rmtree


def split_features(
    count_features: int, feature_path: str, split_dir: Path, current_feature_list: str
) -> None:
    features = pd.read_csv(settings["input_dir"] / Path(feature_path))

    if len(current_feature_list) == 0:
        current_feature_list = []
    else:
        current_feature_list = current_feature_list.split(",")

    list_columns = list(features.columns)
    list_columns.remove("id")
    for el in current_feature_list:
        list_columns.remove(el)
    count_iter = ceil(len(list_columns) / count_features)

    split_dir.mkdir()

    for i in range(count_iter):
        curr_features = list_columns[:count_features]
        list_columns = list_columns[count_features:]

        curr_df = features[["id"] + current_feature_list + curr_features]
        curr_df.to_csv(split_dir / Path(f"split_features_{i+1}.csv"), index=False)
        del curr_df
    del features

    return None


def create_block_var(
    name: str,
    split_dir: Path,
    analytics_path: Path,
    in_name: str,
    out_name: str,
    type_of_model: str,
    current_feature_list: str,
) -> None:
    if len(current_feature_list) == 0:
        current_feature_list = ""
    else:
        current_feature_list = f" -cfl {current_feature_list}"

    files = sorted([file for file in split_dir.iterdir()])
    for file in files:
        full_path = f'{Path("split_features").name}/{Path(file).name}'
        experiment_name = Path(file).name[:-4]
        os.system(
            f"python 3_0_create_block_var.py -feature_path {full_path}\
            -experiment_name {experiment_name} \
            -in_name {in_name} -out_name {out_name} \
            -type_of_model {type_of_model}{current_feature_list} &> file"
        )
        # Path("file").unlink()

        curr_block_var = pd.read_csv(
            analytics_path / Path(f"block_vars_{experiment_name}.csv")
        )
        (analytics_path / Path(f"block_vars_{experiment_name}.csv")).unlink()

        try:
            block_var = pd.concat([block_var, curr_block_var], axis=0).reset_index(
                drop=True
            )
        except NameError:
            block_var = curr_block_var
        block_var = block_var.sort_values(
            by=["confidence_lower"], ascending=False
        ).reset_index(drop=True)
        block_var.to_csv((analytics_path / Path(f"block_vars{name}.csv")), index=False)

        logging.info(f"1. {Path(file).name} - complete")
    return None


# python 3_1_one_factor.py -count_f 5 -feature_path features_contid.csv
# -name lgb_block_vars -type_of_model lgb -in_name data/full_no_tail -out_name data/full_no_tail -d
if __name__ == "__main__":
    settings = get_setting(
        [
            ["-count_f", "Count features in each df", 20],
            ["-feature_path", "Path of feature df", ""],
            ["-name", "Name of experiment", ""],
            ["-type_of_model", "", "lgb"],
            ["-cfl", "Current feature list", ""],
        ]
    )

    split_dir = settings["curr_dir"] / Path("split_features")
    split_features(
        count_features=int(settings["args"].count_f),
        feature_path=settings["args"].feature_path,
        split_dir=split_dir,
        current_feature_list=settings["args"].cfl,
    )
    logging.info(f"0. Split features df into several df - complete")

    in_name, out_name = settings["args"].in_name, settings["args"].out_name
    analytics_path = settings["input_dir"] / Path("analytics")
    name = settings["args"].name
    if name != "":
        name = "_" + name

    create_block_var(
        name,
        split_dir,
        analytics_path,
        in_name,
        out_name,
        settings["args"].type_of_model,
        settings["args"].cfl,
    )
    logging.info(f"2. Create block_var for all features - complete")

    rmtree(split_dir)
