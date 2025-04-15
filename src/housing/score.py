import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from src.housing.logger import Logger


def score(data_path, model_path):
    X_test_prepared = pd.read_csv(os.path.join(data_path, "housing_test_processed.csv"))
    y_test = pd.read_csv(os.path.join(data_path, "housinglabel_test_processed.csv"))
    lg = Logger(
        "./logs/score.log",
        "file reaad successfully from {}".format(data_path),
        "w",
    )
    lg.logging()
    final_model = pickle.load(open(model_path + "/lin_reg.pkl", "rb"))
    lg = Logger(
        "./logs/score.log",
        "lin_reg.pkl model loaded successfully from {}".format(model_path),
        "a",
    )
    lg.logging()
    final_predictions = final_model.predict(X_test_prepared)
    final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse)
    lg = Logger(
        "./logs/score.log",
        "prediction-done! \n final_mse:{}, final_rmse:{}".format(final_mse, final_rmse),
        "a",
    )
    lg.logging()
    print(final_mse, final_rmse)
    print("check logs @ ", lg.filename)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "test_data_path",
#         type=str,
#         help="description of arg1",
#         nargs="?",
#         default="data/processed/test",
#     )
#     parser.add_argument(
#         "stored_model_path",
#         type=str,
#         help="description of arg2",
#         nargs="?",
#         default="artifacts/model",
#     )
#     args = parser.parse_args()

#     config = configparser.ConfigParser()
#     config.read("setup.cfg")

#     arg1 = args.test_data_path or config["DEFAULT"]["test_data_path"]
#     arg2 = args.stored_model_path or config["DEFAULT"]["stored_model_path"]
#     main(arg1, arg2)
