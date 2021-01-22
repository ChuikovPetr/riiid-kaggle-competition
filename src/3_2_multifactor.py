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

# python 3_2_multifactor.py -max_count 15 -count_f 5 -feature_path one_factor_features_contid.csv -name multi_lgb_1 -d
# python 3_2_multifactor.py -max_count_f 130 -count_f 10 -feature_path one_factor_features.csv -name multi_lgb_full_no_tail -in_name data/full_no_tail -out_name data/full_no_tail -d
if __name__ == "__main__":
    settings = get_setting(
        [
            ["-max_count_f", "Maximum count of features in model", 10],
            ["-count_f", "Count features in each df", 20],
            ["-feature_path", "Path of feature df", ""],
            ["-name", "Name of experiment", ""],
            ["-type_of_model", "", "lgb"],
            ["-cfl", "Current feature list", ""],
        ]
    )

    current_feature_list = settings["args"].cfl
    if len(current_feature_list) == 0:
        current_feature_list = []
        cfl = ""
    else:
        current_feature_list = current_feature_list.split(",")
        cfl = f" -cfl " + ",".join(current_feature_list)

    analytics_path = settings["input_dir"] / Path("analytics")

    count_f = settings["args"].count_f
    feature_path = settings["args"].feature_path
    name = settings["args"].name
    type_of_model = settings["args"].type_of_model
    in_name, out_name = settings["args"].in_name, settings["args"].out_name
    for iter in range(int(settings["args"].max_count_f)):
        logging.info(f"Multi factor :: {iter + 1} :: processing")
        if name != "":
            get_name = "_1f_" + name
        else:
            get_name = "_1f"
        os.system(
            f"python 3_1_one_factor.py -count_f {count_f} \
            -feature_path {feature_path} \
            -name {get_name[1:]} \
            -type_of_model {type_of_model}\
            -in_name {in_name} -out_name {out_name}\
            {cfl}"
        )

        block_vars_new = pd.read_csv(analytics_path / f"block_vars{get_name}.csv")
        (analytics_path / f"block_vars{get_name}.csv").unlink()
        try:
            block_vars = (
                pd.concat([block_vars, block_vars_new], axis=0)
                .sort_values(by=["confidence_lower"], ascending=False)
                .reset_index(drop=True)
            )
        except NameError:
            block_vars = block_vars_new

        max_count_vars = block_vars.count_vars.max()
        name_of_the_best = block_vars[
            (block_vars.count_vars == max_count_vars)
        ].new_var.values[0]
        block_vars.loc[
            (block_vars.count_vars == max_count_vars)
            & (block_vars.new_var != name_of_the_best),
            ["best_in_iter"],
        ] = 0
        logging.info(f"Multi factor :: {iter + 1} :: {name_of_the_best} - complete\n")

        current_feature_list = current_feature_list + [name_of_the_best]
        cfl = f" -cfl " + ",".join(current_feature_list)

        if name != "":
            get_name = "_mf_" + name
        else:
            get_name = "_mf"
        block_vars.to_csv(analytics_path / f"block_vars{get_name}.csv", index=False)
