import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
import pickle
import logging
import argparse
from pathlib import Path

from src.utils import get_setting
from ccf.datasets import get_sample_from_population


def prepare_and_save_dataset(df: pd.DataFrame, name: str) -> None:
    df = df.rename(columns={"answered_correctly": "target"})[
        ["id", "target"]
    ].reset_index(drop=True)
    df.to_csv(settings["output_dir"] / Path(f"{name}.csv"), index=False)
    logging.debug(f"Prepare and save {name}; {name}.shape == {df.shape}")

    return None


if __name__ == "__main__":
    settings = get_setting(
        [
            ["-n", "Count of last user activity in for_train dataset", "23"],
            ["-X_n", "Count of last user activity for X dataset", "15"],
            ["-train_val", "Count of last user activity for train_val dataset", "4"],
            ["-val", "Count of last user activity for val dataset", "4"],
            ["-count_x", "Count of obs in train dataset", "None"],
            ["-count_train_val", "Count of obs in train_val dataset", "None"],
            ["-count_val", "Count of obs in val dataset", "None"],
        ]
    )

    vitrine = pd.read_csv(settings["input_dir"] / Path("vitrine.csv"))
    vitrine = vitrine[
        ["id", "answered_correctly", "user_id", "content_type_id", "timestamp"]
    ]
    logging.debug("0. Load vitrine + get necessary fields - complete.")

    vitrine = vitrine[vitrine.content_type_id == 0].reset_index(drop=True)
    vitrine = vitrine.sort_values(by=["user_id", "timestamp"])
    logging.debug("1. Get tasks only + sort vitrine - complete.")

    if eval(settings["args"].n) != None:
        for_train = (
            vitrine.groupby("user_id")
            .tail(int(settings["args"].n))
            .reset_index(drop=True)
        )
    else:
        for_train = vitrine
    logging.debug("2. Create for_train - complete.")
    del vitrine

    logging.debug("3. Create X, train_val, val:")
    val = for_train.groupby("user_id").tail(int(settings["args"].val))
    for_train.drop(val.index, inplace=True)
    val = get_sample_from_population(val, eval(settings["args"].count_val))
    prepare_and_save_dataset(val, "val")

    train_val = for_train.groupby("user_id").tail(int(settings["args"].train_val))
    for_train.drop(train_val.index, inplace=True)
    train_val = get_sample_from_population(
        train_val, eval(settings["args"].count_train_val)
    )
    prepare_and_save_dataset(train_val, "train_val")

    X = get_sample_from_population(for_train, eval(settings["args"].count_x))
    prepare_and_save_dataset(X, "X")

    for_train = pd.concat([X, train_val, val], axis=0).reset_index(drop=True)
    prepare_and_save_dataset(for_train, "for_train")
