import argparse
import configparser
import os

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from housing.logger import Logger


def load_data(train_data_path, test_data_path):
    X_train = pd.read_csv(os.path.join(train_data_path, "housing_train_processed.csv"))
    y_train = pd.read_csv(
        os.path.join(train_data_path, "housinglabel_train_processed.csv")
    )
    X_test = pd.read_csv(os.path.join(test_data_path, "housing_test_processed.csv"))
    y_test = pd.read_csv(
        os.path.join(test_data_path, "housinglabel_test_processed.csv")
    )
    lg = Logger(
        "./logs/train.log",
        "file read successfully",
        "w",
    )
    lg.logging()
    return X_train, y_train, X_test, y_test


def get_path():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "train_data_path",
        type=str,
        help="description of arg1",
        nargs="?",
        default="data/processed/train/",
    )
    parser.add_argument(
        "test_data_path",
        type=str,
        help="description of arg1",
        nargs="?",
        default="data/processed/test/",
    )
    parser.add_argument(
        "stored_model_path",
        type=str,
        help="description of arg2",
        nargs="?",
        default="artifacts/model",
    )
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read("setup.cfg")

    return args
