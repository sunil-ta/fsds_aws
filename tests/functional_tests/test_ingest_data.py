import os
import sys

import pandas as pd

from src.ingest_data import check_nulls

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def test_null_values():
    """
    It checks if there are any null values in the dataframe
    """
    data_path = "/data/processed/"
    test_data = pd.read_csv(os.getcwd() + data_path + "test/housing_test_processed.csv")
    train_data = pd.read_csv(
        os.getcwd() + data_path + "train/housing_train_processed.csv"
    )
    train_flag = check_nulls(train_data)
    test_flag = check_nulls(test_data)
    flag = train_flag and test_flag
    assert flag
