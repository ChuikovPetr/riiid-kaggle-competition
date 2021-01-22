import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import logging
from pathlib import Path
from sklearn.impute import SimpleImputer

from src.utils import get_setting, divide_features, mul_features, get_necess_features
from functools import reduce


def generate_combinations_of_features(features: pd.DataFrame) -> pd.DataFrame:
    # Divide
    params = [
        (
            "rel_user_content_mean_answ",
            "user_mean_answered_correctly",
            "content_mean_answered_correctly",
        ),
        (
            "rel_user_content_he_mean_answ",
            "user_he_mean_answered_correctly",
            "content_he_mean_answered_correctly",
        ),
        (
            "rel_user_content_ema_answ",
            "user_ema_answered_correctly",
            "content_mean_answered_correctly",
        ),
        (
            "rel_user_content_he_ema_answ",
            "user_he_ema_answered_correctly",
            "content_he_mean_answered_correctly",
        ),
    ]
    for p in params:
        divide_features(features, p[0], p[1], p[2], settings)

    # Multiplication
    params = [
        (
            "mul_user_content_mean_answ",
            "user_mean_answered_correctly",
            "content_mean_answered_correctly",
        ),
        (
            "mul_user_content_ema_answ",
            "user_ema_answered_correctly",
            "content_mean_answered_correctly",
        ),
        (
            "mul_user_content_he_mean_answ",
            "user_he_mean_answered_correctly",
            "content_he_mean_answered_correctly",
        ),
        (
            "mul_user_content_he_ema_answ",
            "user_he_ema_answered_correctly",
            "content_he_mean_answered_correctly",
        ),
    ]
    for p in params:
        mul_features(features, p[0], p[1], p[2], settings)

    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features.fillna(0, inplace=True)
    return features


if __name__ == "__main__":
    settings = get_setting([["-cfl", "Current feature list", ""]])

    features = pd.read_csv(settings["input_dir"] / Path("user_vitrine.csv"))
    content_vitrine = pd.read_csv(settings["input_dir"] / Path("content_vitrine.csv"))
    logging.debug("0. Load user_vitrine and content_vitrine - complete.")

    drop_fields = list(
        set(content_vitrine.columns).intersection(
            set(
                ["timestamp", "prior_question_had_explanation", "part", "kmean_cluster"]
            )
        )
    )
    if len(drop_fields) > 0:
        content_vitrine.drop(
            drop_fields, axis=1, inplace=True,
        )
    features = pd.merge(
        features, content_vitrine, left_on="id", right_on="id", how="inner"
    )
    del content_vitrine
    logging.debug(
        f"1. Join user_vitrine and content_vitrine - complete. features.shape == {features.shape}"
    )

    features = generate_combinations_of_features(features)
    logging.debug(
        f"2. Generate combinations of features (and fillna) - complete. features.shape == {features.shape}"
    )

    features = get_necess_features(features, settings)
    if features.shape[1] != 1:
        features.to_csv(settings["output_dir"] / Path("features.csv"), index=False)
    # (settings["input_dir"] / Path("user_vitrine.csv")).unlink()
    # (settings["input_dir"] / Path("content_vitrine.csv")).unlink()
