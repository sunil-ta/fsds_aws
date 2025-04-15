import os
import tarfile

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from scipy.stats import randint
from six.moves import urllib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedShuffleSplit,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

remote_server_uri = "http://0.0.0.0:5002"  # set to your server URI
mlflow.set_tracking_uri(remote_server_uri)  # or set the MLFLOW_TRACKING_URI in the env

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


fetch_housing_data()
print("done.....")
housing = load_housing_data()


train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

housing["income_cat"] = pd.cut(
    housing["median_income"],
    bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
    labels=[1, 2, 3, 4, 5],
)


split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)


train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)


compare_props = pd.DataFrame(
    {
        "Overall": income_cat_proportions(housing),
        "Stratified": income_cat_proportions(strat_test_set),
        "Random": income_cat_proportions(test_set),
    }
).sort_index()

compare_props["Rand.\\%error"] = (
    100 * compare_props["Random"] / compare_props["Overall"] - 100
)
compare_props["Strat.\\%error"] = (
    100 * compare_props["Stratified"] / compare_props["Overall"] - 100
)

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

housing = strat_train_set.copy()
housing.plot(kind="scatter", x="longitude", y="latitude")
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

# column index
col_names = "total_rooms", "total_bedrooms", "population", "households"
rooms_ix, bedrooms_ix, population_ix, households_ix = [
    housing.columns.get_loc(c) for c in col_names
]  # get the column indices


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
        return np.c_[
            X, rooms_per_household, population_per_household, bedrooms_per_room
        ]


attr_adder = CombinedAttributesAdder()
housing_extra_attribs = attr_adder.transform(housing.values)
housing = pd.DataFrame(
    housing_extra_attribs,
    columns=list(housing.columns)
    + ["rooms_per_household", "population_per_household", "bedrooms_per_room"],
    index=housing.index,
)
housing = strat_train_set.drop(
    "median_house_value", axis=1
)  # drop labels for training set
housing_labels = strat_train_set["median_house_value"].copy()
housing_num = housing.drop("ocean_proximity", axis=1)
num_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("attribs_adder", CombinedAttributesAdder()),
        ("std_scaler", StandardScaler()),
    ]
)
housing_tr = num_pipeline.fit_transform(housing_num)
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]
full_pipeline = ColumnTransformer(
    [
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ]
)
housing_prepared = full_pipeline.fit_transform(housing)


def linear_Regression():
    with mlflow.start_run(run_name="Linear Regression", nested=True):
        lin_reg = LinearRegression()
        lin_reg.fit(housing_prepared, housing_labels)
        housing_predictions = lin_reg.predict(housing_prepared)
        lin_mse = mean_squared_error(housing_labels, housing_predictions)
        lin_rmse = np.sqrt(lin_mse)
        lin_mae = mean_absolute_error(housing_labels, housing_predictions)

        mlflow.log_metric(key="mse", value=lin_mse)
        mlflow.log_metric(key="rmse", value=lin_rmse)
        mlflow.log_metric(key="mae", value=lin_mae)
        # mlflow.log_artifact(data_path)
        print("Save to: {}".format(mlflow.get_artifact_uri()))
        mlflow.set_tag("tag1", "Linear Regression")
        mlflow.sklearn.log_model(lin_reg, "model")


def decision_tree():
    with mlflow.start_run(run_name="Decision Tree Regression", nested=True):
        tree_reg = DecisionTreeRegressor(random_state=42)
        tree_reg.fit(housing_prepared, housing_labels)

        housing_predictions = tree_reg.predict(housing_prepared)
        tree_mse = mean_squared_error(housing_labels, housing_predictions)
        tree_rmse = np.sqrt(tree_mse)

        mlflow.log_metric(key="mse", value=tree_mse)
        mlflow.log_metric(key="rmse", value=tree_rmse)
        # mlflow.log_artifact(data_path)
        print("Save to: {}".format(mlflow.get_artifact_uri()))
        mlflow.set_tag("tag1", "Decision Tree Regression")
        mlflow.sklearn.log_model(tree_reg, "model")


param_distribs = {
    "n_estimators": randint(low=1, high=200),
    "max_features": randint(low=1, high=8),
}

param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
]


def randomized_search_cv(forest_reg):
    with mlflow.start_run(run_name="Randomized search cv", nested=True):
        rnd_search = RandomizedSearchCV(
            forest_reg,
            param_distributions=param_distribs,
            n_iter=10,
            cv=5,
            scoring="neg_mean_squared_error",
            random_state=42,
        )
        rnd_search.fit(housing_prepared, housing_labels)
        cvres = rnd_search.cv_results_

        for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
            print(np.sqrt(-mean_score), params)
        for param in cvres["params"]:
            print(param)
            mlflow.log_param(key="parametres", value=param)
            break


def grid_search_cv(forest_reg):
    with mlflow.start_run(run_name="Grid search cv", nested=True):
        # train across 5 folds, that's a total of (12+6)*5=90 rounds of training
        grid_search = GridSearchCV(
            forest_reg,
            param_grid,
            cv=5,
            scoring="neg_mean_squared_error",
            return_train_score=True,
        )
        grid_search.fit(housing_prepared, housing_labels)
        cvres = grid_search.cv_results_
        for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
            print(np.sqrt(-mean_score), params)

        feature_importances = grid_search.best_estimator_.feature_importances_
        extra_attribs = [
            "rooms_per_hhold",
            "pop_per_hhold",
            "bedrooms_per_room",
        ]
        # cat_encoder = cat_pipeline.named_steps["cat_encoder"] # old solution
        cat_encoder = full_pipeline.named_transformers_["cat"]
        cat_one_hot_attribs = list(cat_encoder.categories_[0])
        attributes = num_attribs + extra_attribs + cat_one_hot_attribs
        sorted(zip(feature_importances, attributes), reverse=True)

        final_model = grid_search.best_estimator_

        X_test = strat_test_set.drop("median_house_value", axis=1)
        y_test = strat_test_set["median_house_value"].copy()
        X_test_prepared = full_pipeline.transform(X_test)
        final_predictions = final_model.predict(X_test_prepared)
        final_mse = mean_squared_error(y_test, final_predictions)
        final_rmse = np.sqrt(final_mse)

        mlflow.log_metric(key="mse", value=final_mse)
        mlflow.log_metric(key="rmse", value=final_rmse)
    return final_model


def random_forest_regressor():
    with mlflow.start_run(run_name="Random forest Regression", nested=True):
        forest_reg = RandomForestRegressor(random_state=42)
        print("Starting Randomized Search Cv ml flow")
        randomized_search_cv(forest_reg)
        print("&&&&&&&&&&&&&&")
        print("Starting Grid Search Cv ml flow")
        final_model = grid_search_cv(forest_reg)
        print("&&&&&&&&&&&&&&")
        mlflow.sklearn.log_model(final_model, "model")
        mlflow.set_tag("tag1", "Random Forest Regression")


mlflow.set_experiment("Modeling")
with mlflow.start_run(
    run_name="modeling parent run",
):
    linear_Regression()
    decision_tree()
    with mlflow.start_run(run_name="random forest parent run", nested=True):
        print("starting Random forest ml flow")
        random_forest_regressor()
