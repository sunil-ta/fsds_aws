import os

import pandas as pd
from evidently import *
from evidently.metrics import *
from evidently.presets import *

# Paths to datasets
REFERENCE_DATA_PATH = "/mnt/d/fsds_aws/data/processed/train/housing_train_processed.csv"
CURRENT_DATA_PATH = "/mnt/d/fsds_aws/data/processed/test/housing_test_processed.csv"
PREDICTIONS_PATH_TRAIN = "/mnt/d/fsds_aws/data/processed/train/model_predictions.csv"
PREDICTIONS_PATH_TEST = "/mnt/d/fsds_aws/data/processed/test/model_predictions.csv"
REPORTS_DIR = "/mnt/d/fsds_aws/reports"

# Ensure the reports directory exists
os.makedirs(REPORTS_DIR, exist_ok=True)

print("Loading datasets...")
reference_data = pd.read_csv(REFERENCE_DATA_PATH)
current_data = pd.read_csv(CURRENT_DATA_PATH)
predictions_train = pd.read_csv(PREDICTIONS_PATH_TRAIN)
predictions_test = pd.read_csv(PREDICTIONS_PATH_TEST)
print("Loading Done.")


print("Generating Data Drift Report...")
data_drift_report_path = os.path.join(REPORTS_DIR, "data_drift_report.html")
data_drift_report = Report([DataDriftPreset()])
data_drift_report.run(
    reference_data=reference_data, current_data=current_data
).save_html(data_drift_report_path)
print(f"Data Drift Report saved to {data_drift_report_path}")


print("Generating Data Summary Report...")
data_summary_report_path = os.path.join(REPORTS_DIR, "data_summary_report.html")
data_summary_report = Report([DataSummaryPreset()])
data_summary_report.run(
    reference_data=reference_data, current_data=current_data
).save_html(data_summary_report_path)
print(f"Data Summary Report saved to {data_summary_report_path}")


print("Generating Text Evals Report...")
text_evals_path = os.path.join(REPORTS_DIR, "text_evals.html")
text_evals = Report([TextEvals()])
text_evals.run(reference_data=reference_data, current_data=current_data).save_html(
    text_evals_path
)
print(f"Data Summary Report saved to {text_evals_path}")


print("Generating ValueStats Report...")
cols = [
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
    "rooms_per_household",
    "bedrooms_per_room",
    "population_per_household",
]
vals = [ValueStats(column=i) for i in cols]
value_stat_report_path = os.path.join(REPORTS_DIR, "value_stat_report.html")
value_stat_report = Report(vals)
value_stat_report.run(
    reference_data=reference_data, current_data=current_data
).save_html(text_evals_path)
print(f"Data ValueStats saved to {value_stat_report_path}")


print("Generating Column Analysis Report...")
column_report_path = os.path.join(REPORTS_DIR, "column_report_path.html")
column_report = Report(
    metrics=[
        ColumnCount(),
        RowCount(),
        DatasetMissingValueCount(),
        DriftedColumnsCount(),
        ConstantColumnsCount(),
        DuplicatedColumnsCount(),
        DuplicatedRowCount(),
        EmptyColumnsCount(),
        EmptyRowsCount(),
    ]
)
column_report.run(reference_data=reference_data, current_data=current_data).save_html(
    column_report_path
)
print(f"Column Analysis Report saved to {column_report_path}")

print("Checking for missing values...")
reference_missing = reference_data.isnull().sum()
current_missing = current_data.isnull().sum()

print("Missing values in reference data:")
print(reference_missing)

print("Missing values in current data:")
print(current_missing)


print("Generating Model Performance Reports...")

y_true_train = predictions_train["actual"]
y_true_test = predictions_train["actual"]
for model_name in predictions_train.columns[1:]:
    print(f"Generating performance report for {model_name}...")
    y_pred_train = predictions_train[model_name]
    y_pred_test = predictions_test[model_name]

    performance_data_train = pd.DataFrame(
        {"target": y_true_train, "prediction": y_pred_train}
    )

    performance_data_test = pd.DataFrame(
        {"target": y_true_test, "prediction": y_pred_test}
    )

    performance_report_path = os.path.join(
        REPORTS_DIR, f"{model_name}_performance_report.html"
    )
    data_definition = DataDefinition(
        numerical_columns=["target", "prediction"],
        regression=[Regression(target="target", prediction="prediction")],
    )

    dataset_train = Dataset.from_pandas(
        performance_data_train, data_definition=data_definition
    )
    dataset_test = Dataset.from_pandas(
        performance_data_test, data_definition=data_definition
    )

    report = Report(metrics=[RegressionPreset()])
    report.run(reference_data=dataset_train, current_data=dataset_test).save_html(
        performance_report_path
    )
