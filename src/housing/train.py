import argparse
import configparser
import logging
import os
import pickle
import sys

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from housing.helper import get_path, load_data
from housing.logger import Logger


def linear_regression(X_train, y_train, output_model_path, run_id):
    with mlflow.start_run(run_id=run_id):
        with mlflow.start_run(run_name="Linear Regression", nested=True):
            lin_reg = LinearRegression()
            lin_reg.fit(X_train, y_train)

            predictions = lin_reg.predict(X_train)
            mse = mean_squared_error(y_train, predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_train, predictions)

            mlflow.log_metric("mse", mse)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)

            # output_model_path = 'models/linear_regression/'
            os.makedirs(output_model_path, exist_ok=True)
            with open(os.path.join(output_model_path, "lin_reg.pkl"), "wb") as f:
                pickle.dump(lin_reg, f)

            mlflow.sklearn.log_model(lin_reg, "model")

            lg = Logger(
                "./logs/train.log",
                "lin_reg.pkl model fit successfully and stored to {}".format(
                    mlflow.get_artifact_uri()
                ),
                "a",
            )
            lg.logging()


def decision_tree(X_train, y_train, output_model_path, run_id):
    with mlflow.start_run(run_id=run_id):
        with mlflow.start_run(run_name="Decision Tree Regression", nested=True):
            tree_reg = DecisionTreeRegressor(random_state=42)
            tree_reg.fit(X_train, y_train)

            predictions = tree_reg.predict(X_train)
            mse = mean_squared_error(y_train, predictions)
            rmse = np.sqrt(mse)

            mlflow.log_metric("mse", mse)
            mlflow.log_metric("rmse", rmse)

            # output_model_path = 'models/decision_tree/'
            os.makedirs(output_model_path, exist_ok=True)
            with open(os.path.join(output_model_path, "decision_tree.pkl"), "wb") as f:
                pickle.dump(tree_reg, f)

            mlflow.sklearn.log_model(tree_reg, "model")
            # logger.info(f"Decision Tree model saved to: {mlflow.get_artifact_uri()}")
            lg = Logger(
                "./logs/train.log",
                "decision_tree.pkl model fit successfully and stored to {}".format(
                    mlflow.get_artifact_uri()
                ),
                "a",
            )
            lg.logging()


def random_forest(X_train, y_train, output_model_path, run_id):
    with mlflow.start_run(run_id=run_id):
        with mlflow.start_run(run_name="Random Forest", nested=True):
            forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
            forest_reg.fit(X_train, y_train)

            predictions = forest_reg.predict(X_train)
            mse = mean_squared_error(y_train, predictions)
            rmse = np.sqrt(mse)

            mlflow.log_metric("mse", mse)
            mlflow.log_metric("rmse", rmse)

            # output_model_path = 'models/random_forest/'
            os.makedirs(output_model_path, exist_ok=True)
            with open(os.path.join(output_model_path, "random_forest.pkl"), "wb") as f:
                pickle.dump(forest_reg, f)

            mlflow.sklearn.log_model(forest_reg, "model")
            # logger.info(f"Random Forest model saved to: {mlflow.get_artifact_uri()}")
            lg = Logger(
                "./logs/train.log",
                "random_forest.pkl model fit successfully and stored to {}".format(
                    mlflow.get_artifact_uri()
                ),
                "a",
            )
            lg.logging()


def randomized_search_cv(forest_reg, X_train, y_train, output_model_path, run_id):
    with mlflow.start_run(run_id=run_id):
        with mlflow.start_run(run_name="Randomized Search CV", nested=True):
            param_distribs = {
                "n_estimators": randint(10, 200),
                "max_features": randint(1, 8),
            }

            rnd_search = RandomizedSearchCV(
                forest_reg,
                param_distributions=param_distribs,
                n_iter=10,
                cv=5,
                scoring="neg_mean_squared_error",
                random_state=42,
                n_jobs=-1,
            )
            rnd_search.fit(X_train, y_train)

            best_model = rnd_search.best_estimator_
            best_params = rnd_search.best_params_
            best_score = np.sqrt(-rnd_search.best_score_)

            mlflow.log_params(best_params)
            mlflow.log_metric("best_rmse", best_score)

            # output_model_path = 'models/random_forest_random_cv/'
            os.makedirs(output_model_path, exist_ok=True)
            with open(os.path.join(output_model_path, "random_cv.pkl"), "wb") as f:
                pickle.dump(best_model, f)

            mlflow.sklearn.log_model(best_model, "model")

            lg = Logger(
                "./logs/train.log",
                "random_cv.pkl model fit successfully and stored to {}".format(
                    mlflow.get_artifact_uri()
                ),
                "a",
            )
            lg.logging()


def grid_search_cv(forest_reg, X_train, y_train, output_model_path, run_id):
    with mlflow.start_run(run_id=run_id):
        with mlflow.start_run(run_name="Grid Search CV", nested=True):
            param_grid = [
                {"n_estimators": [10, 50, 100], "max_features": [2, 4, 6, 8]},
                {
                    "bootstrap": [False],
                    "n_estimators": [10, 50],
                    "max_features": [2, 3, 4],
                },
            ]

            grid_search = GridSearchCV(
                forest_reg,
                param_grid,
                cv=5,
                scoring="neg_mean_squared_error",
                return_train_score=True,
                n_jobs=-1,
            )
            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_score = np.sqrt(-grid_search.best_score_)

            mlflow.log_params(best_params)
            mlflow.log_metric("best_rmse", best_score)

            # output_model_path = 'models/random_forest_grid_cv/'
            os.makedirs(output_model_path, exist_ok=True)
            with open(os.path.join(output_model_path, "grid_cv.pkl"), "wb") as f:
                pickle.dump(best_model, f)

            mlflow.sklearn.log_model(best_model, "model")

            lg = Logger(
                "./logs/train.log",
                "grid_cv.pkl model fit successfully and stored to {}".format(
                    mlflow.get_artifact_uri()
                ),
                "a",
            )
            lg.logging()


def train(args):
    X_train, y_train, X_test, y_test = load_data(
        args.train_data_path, args.test_data_path
    )
    output_model_path = args.stored_model_path
    run_id = args.run_id
    # mlflow.set_experiment("Modeling")

    lg = Logger(
        "./logs/train.log",
        "Training Linear Regression model...",
        "a",
    )
    lg.logging()
    linear_regression(X_train, y_train, output_model_path, run_id)

    lg = Logger(
        "./logs/train.log",
        "Training Decision Tree model...",
        "a",
    )
    lg.logging()
    decision_tree(X_train, y_train, output_model_path, run_id)

    lg = Logger(
        "./logs/train.log",
        "Training Random Forest model...",
        "a",
    )
    lg.logging()
    random_forest(X_train, y_train, output_model_path, run_id)

    forest_reg = RandomForestRegressor(random_state=42)
    lg = Logger(
        "./logs/train.log",
        "Running Randomized Search CV for Random Forest...",
        "a",
    )
    lg.logging()
    randomized_search_cv(forest_reg, X_train, y_train, output_model_path, run_id)

    lg = Logger(
        "./logs/train.log",
        "Running Grid Search CV for Random Forest...",
        "a",
    )
    lg.logging()
    grid_search_cv(forest_reg, X_train, y_train, output_model_path, run_id)
