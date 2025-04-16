import os
import pickle
import sys

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from housing.helper import get_path, load_data
from housing.logger import Logger


def evaluate_and_log(model_name, model_path, X_test, y_test):
    try:
        with mlflow.start_run(run_id=sys.argv[1]):
            with mlflow.start_run(run_name=f"{model_name}_score", nested=True):
                model_file = os.path.join(model_path, f"{model_name}.pkl")
                model = pickle.load(open(model_file, "rb"))

                lg = Logger(
                    "./logs/score.log",
                    f"{model_name} model loaded from {model_file}",
                    "a",
                )
                lg.logging()

                preds = model.predict(X_test)
                mse = mean_squared_error(y_test, preds)
                rmse = np.sqrt(mse)
                lg = Logger(
                    "./logs/score.log",
                    f"{model_name} - MSE: {mse:.4f}, RMSE: {rmse:.4f}",
                    "a",
                )
                lg.logging()

                # with mlflow.start_run(run_name=f"{model_name}_score", nested=True):
                mlflow.log_param("model_name", model_name)
                mlflow.log_metric("mse", mse)
                mlflow.log_metric("rmse", rmse)
                mlflow.sklearn.log_model(model, artifact_path="model")

            print(f"{model_name} - MSE: {mse:.4f}, RMSE: {rmse:.4f}")

    except Exception as e:
        lg = Logger(
            "./logs/score.log",
            f"{model_name} - FAILED: {str(e)}",
            "a",
        )
        lg.logging()


def score(args):
    # args = get_path()
    X_train, y_train, X_test, y_test = load_data(
        args.train_data_path, args.test_data_path
    )
    lg = Logger(
        "./logs/score.log",
        f"Scoring started using test data from {args.test_data_path}",
        "w",
    )
    lg.logging()

    # mlflow.set_tracking_uri("http://127.0.0.1:5000")  # or your remote URI
    # mlflow.set_experiment("Housing_Price_Prediction_Score")

    model_names = [
        "lin_reg",
        "decision_tree",
        "random_forest",
        "random_cv",
        "grid_cv",
    ]

    for model_name in model_names:
        evaluate_and_log(model_name, args.stored_model_path, X_test, y_test)

    print("✅ Scoring complete. Check logs and MLflow UI.")
    print("📍 Logs @")
