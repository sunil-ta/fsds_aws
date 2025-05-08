import os
import pickle
import sys
import warnings

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from housing.helper import get_path, load_data
from housing.logger import Logger

warnings.filterwarnings("ignore")


def evaluate_and_log(
    model_name,
    model_path,
    X_train,
    y_train,
    X_test,
    y_test,
    run_id,
    predictions_df_test,
    predictions_df_train,
):
    try:
        with mlflow.start_run(run_id=run_id):
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

                # Log metrics and model to MLflow
                mlflow.log_param("model_name", model_name)
                mlflow.log_metric("mse", mse)
                mlflow.log_metric("rmse", rmse)
                mlflow.sklearn.log_model(model, artifact_path="model")

                # Add predictions to the DataFrame
                predictions_df_test[model_name] = preds
                predictions_df_train[model_name] = model.predict(X_train)

            print(f"{model_name} - MSE: {mse:.4f}, RMSE: {rmse:.4f}")
            return predictions_df_test, predictions_df_train

    except Exception as e:
        lg = Logger(
            "./logs/score.log",
            f"{model_name} - FAILED: {str(e)}",
            "a",
        )
        lg.logging()


def score(args):
    # Load data
    X_train, y_train, X_test, y_test = load_data(
        args.train_data_path, args.test_data_path
    )
    run_id = args.run_id
    lg = Logger(
        "./logs/score.log",
        f"Scoring started using test data from {args.test_data_path}",
        "w",
    )
    lg.logging()

    # Initialize predictions DataFrame
    predictions_df_test = pd.DataFrame()
    predictions_df_test["actual"] = y_test

    predictions_df_train = pd.DataFrame()
    predictions_df_train["actual"] = y_train

    # Model names
    model_names = [
        "lin_reg",
        "decision_tree",
        "random_forest",
        "random_cv",
        "grid_cv",
    ]

    # Evaluate and log each model
    for model_name in model_names:
        predictions_df_test, predictions_df_train = evaluate_and_log(
            model_name,
            args.stored_model_path,
            X_train,
            y_train,
            X_test,
            y_test,
            run_id,
            predictions_df_test,
            predictions_df_train,
        )

    # Save predictions to a CSV file
    predictions_path_train = os.path.join(args.train_data_path, "model_predictions.csv")
    predictions_df_train.to_csv(predictions_path_train, index=False)
    lg = Logger(
        "./logs/score.log",
        f"Predictions saved to {predictions_df_train}",
        "a",
    )
    lg.logging()

    predictions_path_test = os.path.join(args.test_data_path, "model_predictions.csv")
    predictions_df_test.to_csv(predictions_path_test, index=False)
    lg = Logger(
        "./logs/score.log",
        f"Predictions saved to {predictions_path_test}",
        "a",
    )
    lg.logging()

    print("✅ Scoring complete. Check logs and MLflow UI.")
    print("📍 Predictions saved @")
