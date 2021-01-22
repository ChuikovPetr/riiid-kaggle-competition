import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import logging
from pathlib import Path
from sklearn.impute import SimpleImputer

from src.utils import get_setting, divide_features, get_necess_features


def save_features(
    features: pd.DataFrame, vitrine: pd.DataFrame, path: Path, message: str
) -> None:
    features = concat_id(vitrine, features)
    features = get_necess_features(features, settings)
    if features.shape[1] == 1:
        return None

    features = limit_sample_and_fillna(features)

    features.to_csv(settings["output_dir"] / path, index=False)
    del features
    logging.debug(message)

    return None


def ema_window(df_sub: pd.DataFrame, subfield: str = "lag"):
    N = len(df_sub)
    alpha = 2 / (N + 1)
    df_sub["exp"] = df_sub[subfield].ewm(alpha=alpha).mean()
    return df_sub


def compute_cumulative_features(
    df: pd.DataFrame,
    agg_fields: List[str],
    target: str = "answered_correctly",
    prename: str = "user",
    fl_lag: bool = True,
):
    if fl_lag:
        df["lag"] = df.groupby(agg_fields)[target].shift()
    else:
        df["lag"] = df[target]

    cum_sum = df[agg_fields + ["lag"]].groupby(agg_fields)["lag"].cumsum().to_frame()
    cum_count = (
        df[agg_fields + ["lag"]].groupby(agg_fields)["lag"].cumcount().to_frame()
    )

    cum = pd.concat([cum_sum, cum_count], axis=1)
    cum.columns = ["cumsum", "cumcount"]
    del cum_sum
    del cum_count

    cum["mean"] = cum["cumsum"] / cum["cumcount"]

    if not fl_lag:
        cum["cumcount"] += 1
    cum = cum[["cumsum", "cumcount", "mean"]]

    cum.columns = [
        f"{prename}_sum_{target}",
        f"{prename}_count_{target}",
        f"{prename}_mean_{target}",
    ]

    df = df.groupby(agg_fields).apply(ema_window)
    cum[f"{prename}_ema_{target}"] = df["exp"]
    df = df.drop(["lag", "exp"], axis=1)

    # cum.replace([np.inf, -np.inf], np.nan, inplace=True)
    return cum


def cumcount_unique(df):
    series = df["lag"]
    l = list(series)

    already_l = []
    count_unique = 0
    res = []

    for el in l:
        if not el in already_l and el == el:
            count_unique += 1
            already_l.append(el)
            res.append(count_unique)

        elif not el in already_l:
            already_l.append(el)
            res.append(count_unique)

        else:
            res.append(count_unique)
    return pd.DataFrame(data={"col": res})


def compute_cumulative_count_features(
    df: pd.DataFrame,
    agg_fields: List[str],
    target: str = "answered_correctly",
    prename: str = "user",
    fl_lag: bool = True,
):
    if fl_lag:
        df["lag"] = df.groupby(agg_fields)[target].shift()
    else:
        df["lag"] = df[target]

    cum = df.groupby(agg_fields).apply(cumcount_unique).reset_index(drop=True)
    cum.index = df.index
    cum.columns = [f"{prename}_ucount_{target}"]

    return cum


def concat_id(vitrine: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    columns = list(features.columns)
    features.loc[:, "id"] = vitrine["id"]
    features = features[["id"] + columns]
    return features


def limit_sample_and_fillna(features: pd.DataFrame) -> pd.DataFrame:
    # Limit
    for_train = pd.read_csv(settings["input_dir"] / Path("for_train.csv"))[["id"]]
    features = pd.merge(
        for_train, features, left_on="id", right_on="id", how="inner"
    ).reset_index(drop=True)
    del for_train
    features.replace([np.inf, -np.inf], np.nan, inplace=True)

    # fillna
    imputer = SimpleImputer(strategy="median")

    X = pd.read_csv(settings["input_dir"] / Path("X.csv"))[["id"]]
    X_features = pd.merge(
        X, features, left_on="id", right_on="id", how="inner"
    ).reset_index(drop=True)
    imputer.fit(X_features)
    del X
    del X_features

    features_tr = pd.DataFrame(imputer.transform(features), columns=features.columns)
    del features
    return features_tr


def generate_lecture_features(vitrine: pd.DataFrame) -> None:
    type_of_params = [
        "content_type_id",
        "concept",
        "intention",
        "solving_question",
        "starter",
    ]
    aggr_params = [
        (["user_id", "part"], "user_part"),  # (aggr_list, name)
        (["user_id", "kmean_cluster"], "user_kmean_cluster"),
        (["user_id"], "user"),
    ]

    for type_of in type_of_params:

        type_of_name = ""
        if type_of != "content_type_id":
            type_of_name = f"_{type_of}"

        for aggr in aggr_params:
            name = f"lect_{aggr[1]}{type_of_name}"

            cum = vitrine.groupby(aggr[0])[type_of].agg(["cumsum"])
            cum.columns = [name]
            try:
                lect_features = pd.concat([lect_features, cum], axis=1)
            except:
                lect_features = cum
    lect_features.fillna(0, inplace=True)

    save_features(
        lect_features,
        vitrine,
        Path("lect_features.csv"),
        "1. Generate lect_features - complete.",
    )

    return None


def generate_user_correctly(vitrine: pd.DataFrame) -> None:
    user_correctly = compute_cumulative_features(
        vitrine, ["user_id"], target="answered_correctly", prename="user"
    )
    save_features(
        user_correctly,
        vitrine,
        Path("user_correctly.csv"),
        "2. Generate user_correctly - complete.",
    )

    user_he_correctly = compute_cumulative_features(
        vitrine,
        ["user_id", "prior_question_had_explanation"],
        target="answered_correctly",
        prename="user_he",
    )
    save_features(
        user_he_correctly,
        vitrine,
        Path("user_he_correctly.csv"),
        "3. Generate user_he_correctly - complete.",
    )

    vitrine["attempt_no"] = 1
    user_attempt_1 = (
        vitrine[["user_id", "content_id", "answered_correctly"]]
        .groupby(["user_id", "content_id"])["answered_correctly"]
        .cumsum()
        .to_frame()
    )
    user_attempt_2 = (
        vitrine[["user_id", "content_id", "attempt_no"]]
        .groupby(["user_id", "content_id"])["attempt_no"]
        .cumsum()
        .to_frame()
    )
    user_attempt = pd.concat([user_attempt_1, user_attempt_2], axis=1)
    user_attempt.columns = ["attempt_no_sum", "attempt_no_count"]
    user_attempt.attempt_no_sum = (
        user_attempt.attempt_no_sum - vitrine.answered_correctly
    )
    user_attempt.attempt_no_count = user_attempt.attempt_no_count - 1
    user_attempt["attempt_no_mean"] = user_attempt["attempt_no_sum"] / np.where(
        user_attempt["attempt_no_count"] > 0, user_attempt["attempt_no_count"], 1
    )
    del user_attempt_1
    del user_attempt_2
    vitrine.drop(["attempt_no"], axis=1, inplace=True)
    save_features(
        user_attempt,
        vitrine,
        Path("user_attempt.csv"),
        "4. Generate user_attempt - complete.",
    )

    return None


def generate_user_time(vitrine: pd.DataFrame) -> None:
    user_time = compute_cumulative_features(
        vitrine,
        ["user_id"],
        target="prior_question_elapsed_time",
        prename="user",
        fl_lag=False,
    )
    save_features(
        user_time, vitrine, Path("user_time.csv"), "5. Generate user_time - complete.",
    )

    user_he_time = compute_cumulative_features(
        vitrine,
        ["user_id", "prior_question_had_explanation"],
        target="prior_question_elapsed_time",
        prename="user_he",
        fl_lag=False,
    )
    save_features(
        user_he_time,
        vitrine,
        Path("user_he_time.csv"),
        "6. Generate user_he_time - complete.",
    )

    return None


def generate_user_content(vitrine: pd.DataFrame) -> None:
    user_content = compute_cumulative_count_features(
        vitrine, agg_fields=["user_id"], prename="user", target="part"
    )

    user_content_2 = compute_cumulative_count_features(
        vitrine, agg_fields=["user_id"], prename="user", target="kmean_cluster"
    )
    user_content = pd.concat([user_content, user_content_2], axis=1)
    del user_content_2

    user_content_3 = compute_cumulative_count_features(
        vitrine,
        agg_fields=["user_id", "prior_question_had_explanation"],
        prename="user_he",
        target="part",
    )
    user_content = pd.concat([user_content, user_content_3], axis=1)
    del user_content_3

    user_content_4 = compute_cumulative_count_features(
        vitrine,
        agg_fields=["user_id", "prior_question_had_explanation"],
        prename="user_he",
        target="kmean_cluster",
    )
    user_content = pd.concat([user_content, user_content_4], axis=1)
    del user_content_4

    save_features(
        user_content,
        vitrine,
        Path("user_content.csv"),
        "7. Generate user_content - complete.",
    )

    return None


def generate_timestamp_features(vitrine: pd.DataFrame) -> None:
    for i in range(3):
        vitrine[f"lag_timestamp_{i+1}"] = vitrine.groupby(["user_id"])[
            "timestamp"
        ].shift(i + 1)
        vitrine[f"abs_chng_timestamp_{i+1}"] = abs(
            vitrine["timestamp"] - vitrine[f"lag_timestamp_{i+1}"]
        )
        vitrine[f"rel_chng_timestamp_{i+1}"] = (
            vitrine[f"abs_chng_timestamp_{i+1}"] / vitrine["timestamp"]
        )
    logging.debug("Time lag - complete.")
    # Mean and EMA
    ts_vitrine = vitrine.loc[:, ["user_id", "abs_chng_timestamp_1"]]
    cum = compute_cumulative_features(
        ts_vitrine,
        ["user_id"],
        target="abs_chng_timestamp_1",
        prename="ts",
        fl_lag=False,
    )
    del ts_vitrine
    logging.debug("Cum - complete.")

    features = cum
    nec_fields = (
        [f"lag_timestamp_{i+1}" for i in range(3)]
        + [f"abs_chng_timestamp_{i+1}" for i in range(3)]
        + [f"rel_chng_timestamp_{i+1}" for i in range(3)]
    )
    ts_vitrine = vitrine[nec_fields]
    vitrine.drop(
        nec_fields, axis=1, inplace=True,
    )
    features = pd.concat([features, ts_vitrine], axis=1)
    del ts_vitrine
    logging.debug(f"Collect features - complete.{vitrine.columns}")

    # rel timestamp
    features["rel_mean_chng_timestamp"] = (
        features["abs_chng_timestamp_1"] / features["ts_mean_abs_chng_timestamp_1"]
    )

    features["rel_ema_chng_timestamp"] = (
        features["abs_chng_timestamp_1"] / features["ts_ema_abs_chng_timestamp_1"]
    )
    logging.debug("Relative - complete.")

    save_features(
        features,
        vitrine,
        Path("user_timestamp.csv"),
        "8. Generate user_timestamp - complete.",
    )
    return None


def generate_self_rel_features(features: pd.DataFrame) -> pd.DataFrame:
    # Relation features
    params = [
        (
            "rel_user_he_mean_answ",
            "user_he_mean_answered_correctly",
            "user_mean_answered_correctly",
        ),
        (
            "rel_user_he_ema_answ",
            "user_he_ema_answered_correctly",
            "user_ema_answered_correctly",
        ),
        (
            "rel_user_he_mean_pqet",
            "user_he_mean_prior_question_elapsed_time",
            "user_mean_prior_question_elapsed_time",
        ),
        (
            "rel_user_he_ema_pqet",
            "user_he_ema_prior_question_elapsed_time",
            "user_ema_prior_question_elapsed_time",
        ),
        ("rel_strike", "strike", "user_count_answered_correctly",),
        ("rel_strike_part", "strike_part", "user_count_answered_correctly",),
        ("rel_strike_bundle", "strike_bundle", "user_count_answered_correctly",),
        (
            "rel_strike_kmean_cluster",
            "strike_kmean_cluster",
            "user_count_answered_correctly",
        ),
    ]

    for p in params:
        divide_features(features, p[0], p[1], p[2], settings)

    # Relative lect features
    params = [
        ("rel_lect_part", "lect_user_part", "lect_user"),
        ("rel_lect_clu", "lect_user_kmean_cluster", "lect_user"),
    ]
    for p in params:
        divide_features(features, p[0], p[1], p[2], settings)

    for type_of in ["concept", "intention", "solving_question", "starter"]:
        divide_features(
            features,
            f"rel_lect_{type_of}",
            f"lect_user_{type_of}",
            "lect_user",
            settings,
        )

    # Trend features
    params = [
        (
            "trend_user_mean_answ",
            "user_ema_answered_correctly",
            "user_mean_answered_correctly",
        ),
        (
            "trend_user_he_mean_answ",
            "user_he_ema_answered_correctly",
            "user_he_mean_answered_correctly",
        ),
        (
            "trend_user_mean_pqet",
            "user_ema_prior_question_elapsed_time",
            "user_mean_prior_question_elapsed_time",
        ),
        (
            "trend_user_he_mean_pqet",
            "user_he_ema_prior_question_elapsed_time",
            "user_he_mean_prior_question_elapsed_time",
        ),
    ]
    for p in params:
        divide_features(features, p[0], p[1], p[2], settings)

    features.fillna(0, inplace=True)

    logging.debug(
        f"12. Generate self relative features (and fillna) - complete. features.shape == {features.shape}"
    )
    return features


def concatenate_user_vitrine(vitrine: pd.DataFrame) -> None:

    res = vitrine[
        [
            "id",
            "user_id",
            "timestamp",
            "prior_question_elapsed_time",
            "prior_question_had_explanation",
            "part",
            "kmean_cluster",
        ]
    ]
    res = limit_sample_and_fillna(res)
    res["prior_question_had_explanation"] = list(
        map(lambda x: int(x), list(res["prior_question_had_explanation"]))
    )

    for df_path in [
        settings["output_dir"] / Path("lect_features.csv"),
        settings["output_dir"] / Path("user_correctly.csv"),
        settings["output_dir"] / Path("user_he_correctly.csv"),
        settings["output_dir"] / Path("user_attempt.csv"),
        settings["output_dir"] / Path("user_time.csv"),
        settings["output_dir"] / Path("user_he_time.csv"),
        settings["output_dir"] / Path("user_content.csv"),
        settings["output_dir"] / Path("user_timestamp.csv"),
        settings["output_dir"] / Path("lag_features.csv"),
        settings["output_dir"] / Path("strike_features.csv"),
    ]:
        try:
            df = pd.read_csv(df_path)
            res = pd.merge(res, df, left_on="id", right_on="id", how="inner")
            del df
            df_path.unlink()
        except FileNotFoundError:
            if settings["args"].cfl == "":
                raise FileNotFoundError

    logging.debug(f"11. Concatenate user_vitrine - complete. res.shape == {res.shape}")

    res = generate_self_rel_features(res)

    res = get_necess_features(res, settings)
    if res.shape[1] == 1:
        return None

    res.to_csv(settings["output_dir"] / Path("user_vitrine.csv"), index=False)
    del res
    logging.debug("13. Save features - complete.")

    return None


def generate_lag_feature(vitrine: pd.DataFrame) -> None:
    vitrine["lag_answ_corr"] = vitrine.groupby(["user_id"])[
        "answered_correctly"
    ].shift()
    vitrine["lag_part"] = vitrine.groupby(["user_id"])["part"].shift()
    vitrine["lag_clu"] = vitrine.groupby(["user_id"])["kmean_cluster"].shift()

    lag_features = vitrine.loc[
        :,
        [
            "prior_question_had_explanation",
            "part",
            "kmean_cluster",
            "lag_answ_corr",
            "lag_part",
            "lag_clu",
        ],
    ]

    lag_features["lag_part_bool"] = lag_features["part"] == lag_features["lag_part"]
    lag_features["lag_clu_bool"] = (
        lag_features["kmean_cluster"] == lag_features["lag_clu"]
    )
    for name in [
        "lag_answ_corr",
        "lag_part_bool",
        "lag_clu_bool",
        "prior_question_had_explanation",
    ]:
        lag_features[name] = list(map(lambda x: bool(x), list(lag_features[name])))

    lag_features["lag_answ_and_part_bool"] = (
        lag_features["lag_part_bool"] & lag_features["lag_answ_corr"]
    )
    lag_features["lag_answ_and_clu_bool"] = (
        lag_features["lag_clu_bool"] & lag_features["lag_answ_corr"]
    )

    lag_features["lag_expl_and_part_bool"] = (
        lag_features["lag_part_bool"] & lag_features["prior_question_had_explanation"]
    )
    lag_features["lag_expl_and_clu_bool"] = (
        lag_features["lag_clu_bool"] & lag_features["prior_question_had_explanation"]
    )

    lag_features["lag_expl_and_part_bool_not_corr"] = (
        lag_features["lag_part_bool"]
        & lag_features["prior_question_had_explanation"]
        & ~lag_features["lag_answ_corr"]
    )
    lag_features["lag_expl_and_clu_bool_not_corr"] = (
        lag_features["lag_clu_bool"]
        & lag_features["prior_question_had_explanation"]
        & ~lag_features["lag_answ_corr"]
    )
    lag_features = lag_features.drop(
        [
            "prior_question_had_explanation",
            "part",
            "kmean_cluster",
            "lag_part",
            "lag_clu",
        ],
        axis=1,
    )

    vitrine = vitrine.drop(["lag_answ_corr", "lag_part", "lag_clu"], axis=1)

    save_features(
        lag_features,
        vitrine,
        Path("lag_features.csv"),
        "9. Generate lag_features - complete.",
    )

    return None


def strike(series: pd.Series) -> pd.DataFrame:
    cumsum = 0
    res = []
    dict_ = series.to_dict()
    answ_dict = dict_["lag"].copy()
    del dict_["lag"]

    keys = list(dict_.keys())
    try:
        cond_dict = dict_[keys[0]]
    except IndexError:
        cond_dict = None

    for ind, val in answ_dict.items():
        if cond_dict == None:
            if val == 0:
                cumsum = 0
            else:
                cumsum += val
        else:
            try:
                if curr_cond == cond_dict[ind]:
                    if val == 0:
                        cumsum = 0
                    else:
                        cumsum += val
                else:
                    cumsum = 0
                curr_cond = cond_dict[ind]
            except NameError:
                curr_cond = cond_dict[ind]
        res.append(cumsum)

    return pd.DataFrame(data={"col": res})


def generate_strike_feature(df: pd.DataFrame) -> None:
    df["lag"] = df.groupby(["user_id"])["answered_correctly"].shift()

    for el in [
        ([], "strike"),
        (["part"], "strike_part"),
        (["bundle_id"], "strike_bundle"),
        (["kmean_cluster"], "strike_kmean_cluster"),
    ]:
        cum = (
            df.groupby(["user_id"])[["lag"] + el[0]]
            .apply(strike)
            .reset_index(drop=True)
        )
        cum.columns = [el[1]]
        try:
            res = pd.concat([res, cum], axis=1)
            del cum
        except:
            res = cum
    df.drop(["lag"], axis=1, inplace=True)

    save_features(
        res,
        vitrine,
        Path("strike_features.csv"),
        "10. Generate strike_features - complete.",
    )
    return None


if __name__ == "__main__":
    settings = get_setting([["-cfl", "Current feature list", ""]])

    vitrine = pd.read_csv(settings["input_dir"] / Path("vitrine.csv"))
    logging.debug("0. vitrine load - complete.")

    generate_lecture_features(vitrine)
    vitrine = vitrine[vitrine.content_type_id == 0].reset_index(drop=True)
    vitrine["prior_question_had_explanation"].fillna(False, inplace=True)

    generate_user_correctly(vitrine)
    generate_user_time(vitrine)
    generate_user_content(vitrine)

    generate_timestamp_features(vitrine)
    generate_lag_feature(vitrine)
    generate_strike_feature(vitrine)

    concatenate_user_vitrine(vitrine)
