import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
import pickle
import logging
import argparse
from pathlib import Path
import random
from typing import List, Dict, Tuple

from src.utils import get_setting
from ccf.datasets import get_sample_from_population


def prepare_and_save_dataset(df: pd.DataFrame, name: str) -> None:
    df = df.rename(columns={"answered_correctly": "target"})[
        ["id", "target"]
    ].reset_index(drop=True)
    df.to_csv(settings["output_dir"] / Path(f"{name}.csv"), index=False)
    logging.debug(f"Prepare and save {name}; {name}.shape == {df.shape}")

    return None


def rand_time(max_time_stamp: int) -> int:
    interval = MAX_TIME_STAMP - max_time_stamp
    rand_time_stamp = random.randint(0, interval)
    return rand_time_stamp


def get_train_val(
    vitrine: pd.DataFrame, val_ratio: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    val_size = round(val_ratio * len(vitrine))
    logging.debug(f"len(vitrine) == {len(vitrine)}; val_size == {val_size}")

    valid = vitrine[-val_size:]
    vitrine = vitrine[:-val_size]

    # check new users and new contents
    new_users = len(valid[~valid.user_id.isin(vitrine.user_id)].user_id.unique())
    valid_question = valid[valid.content_type_id == 0]
    train_question = vitrine[vitrine.content_type_id == 0]
    new_contents = len(
        valid_question[
            ~valid_question.content_id.isin(train_question.content_id)
        ].content_id.unique()
    )
    logging.debug(
        f"{train_question.answered_correctly.mean():.3f} {valid_question.answered_correctly.mean():.3f} {new_users} {new_contents}"
    )
    return vitrine, valid


if __name__ == "__main__":
    settings = get_setting(
        [
            ["-count_train", "Count of obs in train dataset", "None"],
            ["-count_val", "Count of obs in train_val dataset", "None"],
            ["-val_ratio", "Count of obs in val dataset", "0.2"],
        ]
    )

    vitrine = pd.read_csv(settings["input_dir"] / Path("vitrine.csv"))
    vitrine = vitrine[
        [
            "id",
            "answered_correctly",
            "user_id",
            "content_type_id",
            "timestamp",
            "content_id",
        ]
    ]
    vitrine = vitrine[vitrine.content_type_id == 0].reset_index(drop=True)
    logging.debug("0. Load vitrine + get necessary fields - complete.")

    max_timestamp_u = (
        vitrine[["user_id", "timestamp"]]
        .groupby(["user_id"])
        .agg(["max"])
        .reset_index()
    )
    max_timestamp_u.columns = ["user_id", "max_time_stamp"]
    MAX_TIME_STAMP = max_timestamp_u.max_time_stamp.max()
    logging.debug(
        "1. Gets max timestamp for each user and gets global max timestamp - complete."
    )

    max_timestamp_u["rand_time_stamp"] = max_timestamp_u.max_time_stamp.apply(rand_time)
    vitrine = vitrine.merge(max_timestamp_u, on="user_id", how="left")
    vitrine["viretual_time_stamp"] = vitrine.timestamp + vitrine["rand_time_stamp"]
    logging.debug("2. Create virtual time - complete.")

    vitrine = vitrine.sort_values(["viretual_time_stamp", "id"]).reset_index(drop=True)
    logging.debug("3. Sort by viretual_time_stamp and id - complete.")

    vitrine, valid = get_train_val(vitrine, eval(settings["args"].val_ratio))
    logging.debug("4. Gets train/val - complete.")

    vitrine = get_sample_from_population(vitrine, eval(settings["args"].count_train))
    valid = get_sample_from_population(valid, eval(settings["args"].count_val))
    logging.debug(
        "5. Gets random sample - complete.\n"
        + f"train.shape == {vitrine.shape}\n"
        + f"valid.shape == {valid.shape}\n"
    )

    prepare_and_save_dataset(vitrine, "X")
    prepare_and_save_dataset(valid, "val")

    for_train = pd.concat([vitrine, valid], axis=0).reset_index(drop=True)
    prepare_and_save_dataset(for_train, "for_train")
