import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
import pickle
import logging
import argparse
from pathlib import Path

from src.utils import get_setting


def preprocess_train_df(train: pd.DataFrame) -> pd.DataFrame:
    train = train.drop(["user_answer"], axis=1)
    train = train.rename(columns={"row_id": "id"})
    return train


def preprocess_questions_df(questions: pd.DataFrame) -> pd.DataFrame:
    questions = questions.drop(["correct_answer"], axis=1).rename(
        columns={"question_id": "content_id"}
    )
    questions = preprocessing_tags(questions)

    questions["part_bundle_id"] = questions["part"] * 100000 + questions["bundle_id"]
    questions["clu_bundle_id"] = (
        questions["kmean_cluster"] * 100000 + questions["bundle_id"]
    )
    return questions


def preprocess_lectures_df(lectures: pd.DataFrame) -> pd.DataFrame:
    lectures = lectures.rename(columns={"lecture_id": "content_id", "tag": "tags"})

    lectures["type_of"] = lectures["type_of"].replace(
        "solving question", "solving_question"
    )
    lectures = pd.get_dummies(lectures, columns=["type_of"], prefix="", prefix_sep="")

    lectures = preprocessing_tags(lectures)
    return lectures


def encode_tags(df: pd.DataFrame, tags_field: str, ucount_tags: int) -> pd.DataFrame:
    """
    :param df: (tags_field, ...)
    :param tags_field: name of tag field
    :param ucount_tags: count unique tags

    :return: df (tag_0, tag_1, ...) - OneHot encoding
    """
    rows = []
    for tags in df[tags_field]:
        ohe = np.zeros(ucount_tags)
        if str(tags) != "nan" and tags == tags:
            if isinstance(tags, int):
                ohe += np.eye(ucount_tags)[tags]
            else:
                for tag in tags.split():
                    ohe += np.eye(ucount_tags)[int(tag)]
        rows.append(ohe)

    return pd.DataFrame(rows, columns=[f"tag_{i}" for i in range(ucount_tags)]).astype(
        int
    )


def encode_tags_and_parts(questions: pd.DataFrame) -> pd.DataFrame:
    """
    :param questions: df (tags, part, ...)
    :return: df (tag_0, tag_1, ..., tag_m, part_1, ..., part_n) - OneHot encoding
    """
    tags_df = encode_tags(questions, "tags", 188)

    encoder = OneHotEncoder()
    part_one_hot = pd.DataFrame(
        encoder.fit_transform(questions[["part"]]).toarray(),
        columns=[f"part_{i+1}" for i in range(7)],
    )

    tags_df = pd.concat([tags_df, part_one_hot], axis=1)
    return tags_df


def preprocessing_tags(df: pd.DataFrame) -> pd.DataFrame:
    tags_df = encode_tags_and_parts(df)

    with open(settings["model_dir"] / Path("km_17.pkl"), "rb") as f:
        km = pickle.load(f)

    df[f"kmean_cluster"] = km.predict(tags_df.values)
    df = df.drop(["tags"], axis=1)

    return df


if __name__ == "__main__":
    settings = get_setting([["-rev", "Reverse dataset", "False"]])

    train = pd.read_csv(settings["input_dir"] / Path("train.csv"))
    train = preprocess_train_df(train)
    logging.debug("0. Train load - complete.")

    questions = pd.read_csv(settings["input_dir"] / Path("questions.csv"))
    questions = preprocess_questions_df(questions)
    logging.debug("1. Preprocess questions df - complete.")

    lectures = pd.read_csv(settings["input_dir"] / Path("lectures.csv"))
    lectures = preprocess_lectures_df(lectures)
    logging.debug("2. Preprocess lectures df - complete.")

    train_lect = train[train["content_type_id"] == 1].reset_index(drop=True)
    train_quest = train[train["content_type_id"] == 0].reset_index(drop=True)
    del train

    train_lect = pd.merge(
        train_lect,
        lectures,
        left_on=["content_id"],
        right_on=["content_id"],
        how="left",
    )

    train_quest = pd.merge(
        train_quest,
        questions,
        left_on=["content_id"],
        right_on=["content_id"],
        how="left",
    )

    vitrine = pd.concat([train_lect, train_quest], axis=0)
    del train_lect
    del train_quest
    vitrine = vitrine.sort_values(
        by=["user_id", "timestamp"], ascending=not eval(settings["args"].rev)
    ).reset_index(drop=True)
    logging.debug("3. Create vitrine - complete.")

    vitrine.to_csv(settings["output_dir"] / Path("vitrine.csv"), index=False)
    del vitrine
