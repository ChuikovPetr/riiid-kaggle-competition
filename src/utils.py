from pathlib import Path
import argparse
import logging
import pandas as pd
from typing import Tuple, List, Dict


def get_setting(adding_arguments: List[List[str]] = []):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--debug", help="Activate debug regim", action="store_true"
    )
    parser.add_argument(
        "-in_name", help="Input directory", action="store", default="data/base"
    )
    parser.add_argument(
        "-out_name", help="Output directory", action="store", default="data/base"
    )

    for add in adding_arguments:
        parser.add_argument(add[0], help=add[1], default=add[2], action="store")

    args = parser.parse_args()

    curr_dir = Path(__file__).resolve().parent
    input_dir = curr_dir.parent / Path(args.in_name)
    output_dir = curr_dir.parent / Path(args.out_name)
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True)

    if args.debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s :: %(filename)s :: %(funcName)s :: %(message)s",
        )

    settings = {
        "args": args,
        "curr_dir": curr_dir,
        "input_dir": input_dir,
        "output_dir": output_dir,
        "model_dir": curr_dir.parent / Path("saved_models"),
    }
    return settings


def divide_features(
    features: pd.DataFrame, name: str, field_1: str, field_2: str, settings: Dict
) -> None:
    try:
        features[name] = features[field_1] / features[field_2]
    except KeyError:
        if settings["args"].cfl == "":
            raise KeyError


def mul_features(
    features: pd.DataFrame, name: str, field_1: str, field_2: str, settings: Dict
) -> None:
    try:
        features[name] = features[field_1] * features[field_2]
    except KeyError:
        if settings["args"].cfl == "":
            raise KeyError


def get_necess_features(
    features: pd.DataFrame, settings: Dict, id_fields: List[str] = ["id"]
) -> pd.DataFrame:
    if settings["args"].cfl == "":
        return features

    current_feature_list = settings["args"].cfl
    current_feature_list = current_feature_list.split(",")
    cols = set(features.columns)
    current_feature_list = list(cols.intersection(current_feature_list))

    if len(current_feature_list) == 0:
        return features[id_fields]

    features = features[list(set(id_fields + current_feature_list))]
    return features
