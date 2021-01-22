import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import logging
from pathlib import Path
from sklearn.impute import SimpleImputer

from src.utils import get_setting, get_necess_features, divide_features
from functools import reduce


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


def generate_content_answ_features(vitrine: pd.DataFrame) -> pd.DataFrame:
    params = [
        (
            ["content_id"],
            ["content_mean_", "content_sum_", "content_count_", "content_std_",],
        ),
        (
            ["part"],
            [
                "content_part_mean_",
                "content_part_sum_",
                "content_part_count_",
                "content_part_std_",
            ],
        ),
        (
            ["kmean_cluster"],
            [
                "content_cl_mean_",
                "content_cl_sum_",
                "content_cl_count_",
                "content_cl_std_",
            ],
        ),
        (
            ["content_id", "prior_question_had_explanation"],
            [
                "content_he_mean_",
                "content_he_sum_",
                "content_he_count_",
                "content_he_std_",
            ],
        ),
        (
            ["part", "prior_question_had_explanation"],
            [
                "content_he_part_mean_",
                "content_he_part_sum_",
                "content_he_part_count_",
                "content_he_part_std_",
            ],
        ),
        (
            ["kmean_cluster", "prior_question_had_explanation"],
            [
                "content_he_cl_mean_",
                "content_he_cl_sum_",
                "content_he_cl_count_",
                "content_he_cl_std_",
            ],
        ),
    ]

    content_answ = []
    for param in params:
        content_answ.append(
            aggregate(
                vitrine,
                groupby_list=param[0],
                agg_func=["mean", "sum", "count", "var"],
                name_list=param[1],
                target_field="answered_correctly",
            )
        )

    return content_answ


def generate_content_user_features(vitrine: pd.DataFrame) -> pd.DataFrame:

    params = [
        (["content_id"], ["content_user_count_"]),
        (["part"], ["content_user_part_count_"]),
        (["kmean_cluster"], ["content_user_cl_count_"]),
        (["content_id", "prior_question_had_explanation"], ["content_user_he_count_"]),
        (["part", "prior_question_had_explanation"], ["content_user_he_part_count_"]),
        (
            ["kmean_cluster", "prior_question_had_explanation"],
            ["content_user_he_cl_count_"],
        ),
    ]
    content_user = []
    for param in params:
        content_user.append(
            aggregate(
                vitrine,
                groupby_list=param[0],
                agg_func=["count"],
                name_list=param[1],
                target_field="user_id",
            )
        )

    return content_user


def limit_vitrine_for_feature_gen(vitrine: pd.DataFrame) -> pd.DataFrame:
    try:
        train_val_ids = list(
            pd.read_csv(settings["input_dir"] / Path("train_val.csv"))["id"]
        )
        unwanted_ids = train_val_ids
    except FileNotFoundError:
        unwanted_ids = []
    val_ids = list(pd.read_csv(settings["input_dir"] / Path("val.csv"))["id"])
    unwanted_ids = unwanted_ids + val_ids

    vitrine = vitrine[~vitrine.id.isin(unwanted_ids)].reset_index(drop=True)
    return vitrine


def limit_vitrine_for_training(vitrine: pd.DataFrame) -> pd.DataFrame:
    for_train_ids = list(
        pd.read_csv(settings["input_dir"] / Path("for_train.csv"))["id"]
    )

    vitrine = vitrine[vitrine.id.isin(for_train_ids)].reset_index(drop=True)
    return vitrine


def create_and_fillna_content_features(
    features: pd.DataFrame, content_answ: pd.DataFrame, content_user: pd.DataFrame
) -> pd.DataFrame:
    features = features[
        [
            "id",
            "content_id",
            "timestamp",
            "prior_question_had_explanation",
            "part",
            "kmean_cluster",
            "bundle_id",
            "part_bundle_id",
            "clu_bundle_id",
            "task_container_id",
        ]
    ]

    params_content_answer_corr = [
        ["content_id"],
        ["part"],
        ["kmean_cluster"],
        ["content_id", "prior_question_had_explanation"],
        ["part", "prior_question_had_explanation"],
        ["kmean_cluster", "prior_question_had_explanation"],
    ]
    for ind, el in enumerate(params_content_answer_corr):
        content_answ[ind] = get_necess_features(
            content_answ[ind], settings, id_fields=el
        )
        if content_answ[ind].shape[1] > len(el):
            features = pd.merge(
                features, content_answ[ind], left_on=el, right_on=el, how="left"
            )
    logging.debug(f"Merge content_answ - complete. features.shape == {features.shape}")

    params_content_user = [
        ["content_id"],
        ["part"],
        ["kmean_cluster"],
        ["content_id", "prior_question_had_explanation"],
        ["part", "prior_question_had_explanation"],
        ["kmean_cluster", "prior_question_had_explanation"],
    ]
    for ind, el in enumerate(params_content_user):
        content_user[ind] = get_necess_features(
            content_user[ind], settings, id_fields=el
        )
        if content_user[ind].shape[1] > len(el):
            features = pd.merge(
                features, content_user[ind], left_on=el, right_on=el, how="left"
            )
    logging.debug(f"Merge content_user - complete. features.shape == {features.shape}")

    features = get_necess_features(features, settings)
    logging.debug(
        f"Get only necessary features - complete. features.shape == {features.shape}"
    )

    # fillna
    imputer = SimpleImputer(strategy="median")

    X = pd.read_csv(settings["input_dir"] / Path("X.csv"))[["id"]]
    X_features = pd.merge(
        X, features, left_on="id", right_on="id", how="inner"
    ).reset_index(drop=True)
    imputer.fit(X_features)
    del X
    del X_features
    logging.debug(f"Impute X_features - complete. features.shape == {features.shape}")

    features = pd.DataFrame(imputer.transform(features), columns=features.columns)
    return features


def generate_self_rel_features(features: pd.DataFrame) -> pd.DataFrame:
    # Relation features

    # Part content
    params = [
        (
            "rel_content_part_mean_answ",
            "content_part_mean_answered_correctly",
            "content_mean_answered_correctly",
        ),
        (
            "rel_content_he_part_mean_answ",
            "content_he_part_mean_answered_correctly",
            "content_mean_answered_correctly",
        ),
    ]
    for p in params:
        divide_features(features, p[0], p[1], p[2], settings)

    # Cluster content
    params = [
        (
            "rel_content_cl_mean_answ",
            "content_cl_mean_answered_correctly",
            "content_mean_answered_correctly",
        ),
        (
            "rel_content_he_cl_mean_answ",
            "content_he_cl_mean_answered_correctly",
            "content_mean_answered_correctly",
        ),
    ]
    for p in params:
        divide_features(features, p[0], p[1], p[2], settings)

    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features.fillna(0, inplace=True)
    return features


if __name__ == "__main__":
    settings = get_setting([["-cfl", "Current feature list", ""]])

    vitrine = load_and_preprocess_vitrine()
    logging.debug("0. Vitrine loading and preprocessing - complete.")

    vitrine = limit_vitrine_for_feature_gen(vitrine)
    logging.debug("1. Limit vitrine for feature generation - complete.")

    content_answ = generate_content_answ_features(vitrine)
    logging.debug("2. Generate content_answ - complete.")

    content_user = generate_content_user_features(vitrine)
    logging.debug("3. Generate content_user - complete.")

    del vitrine
    vitrine = load_and_preprocess_vitrine()
    vitrine = limit_vitrine_for_training(vitrine)
    logging.debug(
        f"4. Limit vitrine for training - complete. vitrine.shape == {vitrine.shape}"
    )

    vitrine = create_and_fillna_content_features(vitrine, content_answ, content_user)
    logging.debug(
        f"5. Create content features - complete. vitrine.shape == {vitrine.shape}"
    )

    vitrine = generate_self_rel_features(vitrine)
    logging.debug(
        f"6. Generate self relative features (and fillna) - complete. vitrine.shape == {vitrine.shape}"
    )

    vitrine = get_necess_features(vitrine, settings)
    if vitrine.shape[1] > 1:
        vitrine.to_csv(
            settings["output_dir"] / Path("content_vitrine.csv"), index=False
        )
    del vitrine
    logging.debug("7. Save features - complete.")

    # (settings["input_dir"] / Path("for_train.csv")).unlink()
