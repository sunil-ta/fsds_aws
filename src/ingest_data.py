import os
import tarfile

import numpy as np
import pandas as pd
from six.moves import urllib
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

from src.logger import Logger


def check_nulls(data):
    """
    If there are any null values in the data, return False. Otherwise, \
    return True

    :param data: The dataframe you want to check for null values
    :return: A boolean value.
    """
    if bool(data.isnull().values.any()):
        return False
    else:
        return True


def fetch_housing_data(housing_url, housing_path):
    """
    > It downloads the housing.tgz file from the housing_url,
    extracts the housing.csv
    file from the housing.tgz file, and saves the housing.csv
    file in the housing_path
    directory

    :param housing_url: The URL of the housing dataset (defaults to the \
        one hosted by
    the University of California, Irvine)
    :param housing_path: The directory to save the dataset in
    """
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
    print(housing_path)


def load_housing_data(housing_path):
    """
    It loads the housing data from the given path, and returns a Pandas \
        DataFrame object
    containing the data

    :param housing_path: The path to the housing dataset
    :return: A dataframe
    """
    """
    It loads the housing data from the given path, and returns a Pandas \
        DataFrame object
    containing the data

    :param housing_path: The path to the housing dataset
    :return: A dataframe
    """
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)


def ingest(download):
    print(download)
    DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
    HOUSING_PATH = os.path.join(download, "raw")
    HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
    fetch_housing_data(HOUSING_URL, HOUSING_PATH)
    print(HOUSING_PATH)
    lg = Logger(
        "./logs/ingest.logs",
        "Parsed dta path is {} \n Raw hosuing data fetched to {}".format(
            download, HOUSING_PATH
        ),
        "w",
    )
    lg.logging()
    housing = load_housing_data(HOUSING_PATH)
    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
    if check_nulls(housing):
        housing.dropna(axis=0, how="any", inplace=True)
    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

    compare_props = pd.DataFrame(
        {
            "Overall": income_cat_proportions(housing),
            "Stratified": income_cat_proportions(strat_test_set),
            "Random": income_cat_proportions(test_set),
        }
    ).sort_index()
    compare_props["Rand. %error"] = (
        100 * compare_props["Random"] / compare_props["Overall"] - 100
    )
    compare_props["Strat. %error"] = (
        100 * compare_props["Stratified"] / compare_props["Overall"] - 100
    )

    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    housing = strat_train_set.copy()
    # housing.plot(kind="scatter", x="longitude", y="latitude")
    # housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
    print(housing)
    housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
    housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
    housing["population_per_household"] = housing["population"] / housing["households"]

    housing = strat_train_set.drop(
        "median_house_value", axis=1
    )  # drop labels for training set
    housing_labels = strat_train_set["median_house_value"].copy()

    imputer = SimpleImputer(strategy="median")

    housing_num = housing.drop("ocean_proximity", axis=1)

    imputer.fit(housing_num)
    X = imputer.transform(housing_num)

    housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing.index)

    housing_tr["rooms_per_household"] = (
        housing_tr["total_rooms"] / housing_tr["households"]
    )

    housing_tr["bedrooms_per_room"] = (
        housing_tr["total_bedrooms"] / housing_tr["total_rooms"]
    )
    housing_tr["population_per_household"] = (
        housing_tr["population"] / housing_tr["households"]
    )

    housing_cat = housing[["ocean_proximity"]]

    housing_prepared = housing_tr.join(pd.get_dummies(housing_cat, drop_first=True))

    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()

    X_test_num = X_test.drop("ocean_proximity", axis=1)
    X_test_prepared = imputer.transform(X_test_num)
    X_test_prepared = pd.DataFrame(
        X_test_prepared, columns=X_test_num.columns, index=X_test.index
    )
    X_test_prepared["rooms_per_household"] = (
        X_test_prepared["total_rooms"] / X_test_prepared["households"]
    )
    X_test_prepared["bedrooms_per_room"] = (
        X_test_prepared["total_bedrooms"] / X_test_prepared["total_rooms"]
    )
    X_test_prepared["population_per_household"] = (
        X_test_prepared["population"] / X_test_prepared["households"]
    )

    X_test_cat = X_test[["ocean_proximity"]]
    X_test_prepared = X_test_prepared.join(pd.get_dummies(X_test_cat, drop_first=True))

    housing_processed_dir_path = os.path.join(download, "processed")
    train_path = os.path.join(housing_processed_dir_path, "train")
    os.makedirs(train_path, exist_ok=True)
    housing_prepared.to_csv(train_path + "/housing_train_processed.csv")
    os.makedirs(train_path, exist_ok=True)
    housing_labels.to_csv(train_path + "/housinglabel_train_processed.csv")
    lg = Logger(
        "./logs/ingest.logs",
        "Housing Train data Prepared to {}".format(train_path),
        "a",
    )
    lg.logging()
    test_path = os.path.join(housing_processed_dir_path, "test")
    os.makedirs(test_path, exist_ok=True)
    X_test_prepared.to_csv(test_path + "/housing_test_processed.csv")
    os.makedirs(test_path, exist_ok=True)
    y_test.to_csv(test_path + "/housinglabel_test_processed.csv")
    lg = Logger(
        "./logs/ingest.logs",
        "Housing Test data Prepared to {} \n Converted to CSV...Success!".format(
            test_path
        ),
        "a",
    )
    lg.logging()
    print("Converted to CSV...Success!")
    print("check logs @ ", lg.filename)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "download",
#         type=str,
#         help="description of arg1",
#         nargs="?",
#         default="data",
#     )
#     args = parser.parse_args()
#     config = configparser.ConfigParser()
#     config.read("setup.cfg")

#     arg1 = args.download
#     main(arg1)
