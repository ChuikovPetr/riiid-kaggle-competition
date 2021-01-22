import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import logging
from pathlib import Path
from sklearn.impute import SimpleImputer

from src.utils import get_setting
from functools import reduce
import pickle


def load_and_preprocess_vitrine() -> pd.DataFrame:
    vitrine = pd.read_csv(settings["input_dir"] / Path("vitrine.csv"))

    vitrine = vitrine[vitrine.content_type_id == 0].reset_index(drop=True)
    vitrine["prior_question_had_explanation"].fillna(False, inplace=True)
    vitrine["prior_question_had_explanation"] = list(
        map(lambda x: int(x), list(vitrine["prior_question_had_explanation"]))
    )

    return vitrine


def aggregate(
    df_i: pd.DataFrame,
    groupby_list: List[str],
    agg_func: List,
    name_list: List,
    target_field: str,
) -> pd.DataFrame:
    df = df_i.copy()
    df = df.sort_values(by=groupby_list + ["timestamp"])
    df = df[groupby_list + [target_field]].groupby(groupby_list).agg(agg_func)
    df.columns = list(map(lambda x: x + target_field, name_list))
    return df.reset_index()


def get_user_last_f(
    df_i: pd.DataFrame, keys: List[str], nec_field: List[str]
) -> pd.DataFrame:
    df = df_i.copy()

    df = df[keys + list(set(nec_field) - set(["timestamp"])) + ["timestamp"]]
    df = df.sort_values(by=["user_id", "timestamp"])
    df = (
        df.drop_duplicates(subset=keys, keep="last")
        .sort_values(by=keys)
        .reset_index(drop=True)
    )

    return df


def get_answered_as_arr(
    vitrine: pd.DataFrame,
):
    features_user = vitrine.sort_values(by=["user_id", "timestamp"]).reset_index(
        drop=True
    )
    features_user = features_user[["user_id", "answered_correctly"]]

    return (
        features_user.groupby("user_id")["answered_correctly"]
        .apply(np.array)
        .to_frame()
    ).reset_index()


if __name__ == "__main__":
    settings = get_setting()

    vitrine = load_and_preprocess_vitrine()
    logging.debug(
        "0. Vitrine loading and preprocessing (only content_type == 0) - complete."
    )

    result_features = {
        "user_ema_answered_correctly": get_answered_as_arr(
            vitrine,
        ),
        "abs_chng_timestamp": get_user_last_f(
            vitrine, keys=["user_id"], nec_field=["timestamp"]
        ),
        "content_he_mean_answered_correctly": aggregate(
            vitrine,
            groupby_list=["content_id", "prior_question_had_explanation"],
            agg_func=["sum", "count"],
            name_list=["content_he_sum_", "content_he_count_"],
            target_field="answered_correctly",
        ),
        "content_he_part_sum_answered_correctly": aggregate(
            vitrine,
            groupby_list=["part", "prior_question_had_explanation"],
            agg_func=["sum"],
            name_list=["content_he_part_sum_"],
            target_field="answered_correctly",
        ),
    }
    logging.debug("1. Create result_features - complete.")

    with open(settings["output_dir"] / Path("result_features.pkl"), "wb") as f:
        pickle.dump(result_features, f, protocol=2)
    logging.debug("2. Save result_features - complete.")
