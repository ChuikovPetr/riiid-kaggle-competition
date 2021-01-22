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


def get_kt_set(
    for_train: pd.DataFrame, ratio: float, last_count: int, sample_size: int
) -> pd.DataFrame:
    # 1. Берем подмножество user_id -> T_set
    if ratio != None:
        unique_user = len(for_train["user_id"].unique())
        unique_user_T = int(ratio * unique_user)
        logging.debug(
            f"unique_user == {unique_user}, unique_user_remain == {unique_user-unique_user_T}, unique_user_T == {unique_user_T}"
        )
        T_set = get_sample_from_population(
            for_train, unique_user_T, target_field="user_id"
        )
    else:
        T_set = for_train

    T_set = T_set.sort_values(by=["user_id", "timestamp"])
    # 2. Берем tail_k(T_set) -> KT_set
    if last_count != None:
        KT_set = T_set.groupby("user_id").tail(int(last_count)).reset_index(drop=True)
    else:
        KT_set = T_set

    # 3. Ограничиваем выборку при необходимости
    logging.debug(f"KT_set_len == {len(KT_set)}, sample_size == {sample_size}")
    KT_set_sample = get_sample_from_population(KT_set, sample_size)
    return KT_set_sample


if __name__ == "__main__":
    settings = get_setting(
        [
            ["-n", "Count of last user activity in for_train dataset", 23],
            ["-ratio_train_val", "", 0.1],
            ["-ratio_val", "", 0.2],
            ["-train_val", "Count of last user activity for val dataset", 4],
            ["-val", "Count of last user activity for val dataset", 4],
            ["-count_x", "Count of obs in train dataset", None],
            ["-count_train_val", "Count of obs in train_val dataset", None],
            ["-count_val", "Count of obs in val dataset", None],
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
    val = get_kt_set(
        for_train,
        ratio=eval(settings["args"].ratio_val),
        last_count=eval(settings["args"].val),
        sample_size=eval(settings["args"].count_val),
    )
    prepare_and_save_dataset(val, "val")
    for_train = for_train[~for_train.user_id.isin(list(val.user_id))].reset_index(
        drop=True
    )

    train_val = get_kt_set(
        for_train,
        ratio=eval(settings["args"].ratio_train_val),
        last_count=eval(settings["args"].train_val),
        sample_size=eval(settings["args"].count_train_val),
    )
    prepare_and_save_dataset(train_val, "train_val")
    for_train = for_train[~for_train.user_id.isin(list(train_val.user_id))].reset_index(
        drop=True
    )

    X = get_kt_set(
        for_train,
        ratio=None,
        last_count=None,
        sample_size=eval(settings["args"].count_x),
    )
    prepare_and_save_dataset(X, "X")

    for_train = pd.concat([X, train_val, val], axis=0).reset_index(drop=True)
    prepare_and_save_dataset(for_train, "for_train")
