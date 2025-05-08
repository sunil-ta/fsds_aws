import os
import warnings

import pandas as pd
from evidently import DataDefinition, Dataset, Recsys, Regression, Report
from evidently.metrics import (
    ColumnCount,
    ConstantColumnsCount,
    DatasetMissingValueCount,
    DriftedColumnsCount,
    DuplicatedColumnsCount,
    DuplicatedRowCount,
    EmptyColumnsCount,
    EmptyRowsCount,
    RowCount,
)
from evidently.presets import (
    DataDriftPreset,
    DataSummaryPreset,
    RegressionPreset,
    TextEvals,
    ValueStats,
)

from housing.logger import Logger

warnings.filterwarnings("ignore")


class EvidentlyReportGenerator:
    def __init__(
        self, reference_path, current_path, pred_train_path, pred_test_path, reports_dir
    ):
        self.reference_path = reference_path
        self.current_path = current_path
        self.pred_train_path = pred_train_path
        self.pred_test_path = pred_test_path
        self.reports_dir = reports_dir
        os.makedirs(self.reports_dir, exist_ok=True)

        self.logger = Logger(
            "./logs/model_monitoring.log", "Initialized EvidentlyReportGenerator", "w"
        )
        self.logger.logging()

        self.reference_data = pd.read_csv(self.reference_path)
        self.current_data = pd.read_csv(self.current_path)
        self.predictions_train = pd.read_csv(self.pred_train_path)
        self.predictions_test = pd.read_csv(self.pred_test_path)

    def generate_data_drift_report(self):
        path = os.path.join(self.reports_dir, "data_drift_report.html")
        report = Report([DataDriftPreset()])
        report.run(
            reference_data=self.reference_data, current_data=self.current_data
        ).save_html(path)

        lg = Logger(
            "./logs/model_monitoring.log", f"Data Drift Report saved to {path}", "a"
        )
        lg.logging()

    def generate_data_summary_report(self):
        path = os.path.join(self.reports_dir, "data_summary_report.html")
        report = Report([DataSummaryPreset()])
        report.run(
            reference_data=self.reference_data, current_data=self.current_data
        ).save_html(path)

        lg = Logger(
            "./logs/model_monitoring.log", f"Data Summary Report saved to {path}", "a"
        )
        lg.logging()

    def generate_text_eval_report(self):
        path = os.path.join(self.reports_dir, "text_evals.html")
        report = Report([TextEvals()])
        report.run(
            reference_data=self.reference_data, current_data=self.current_data
        ).save_html(path)

        lg = Logger(
            "./logs/model_monitoring.log", f"Text Evals Report saved to {path}", "a"
        )
        lg.logging()

    def generate_value_stats_report(self):
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
        metrics = [ValueStats(column=c) for c in cols]
        path = os.path.join(self.reports_dir, "value_stat_report.html")
        report = Report(metrics)
        report.run(
            reference_data=self.reference_data, current_data=self.current_data
        ).save_html(path)

        lg = Logger(
            "./logs/model_monitoring.log", f"Value Stats Report saved to {path}", "a"
        )
        lg.logging()

    def generate_column_analysis_report(self):
        path = os.path.join(self.reports_dir, "column_report.html")
        report = Report(
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
        report.run(
            reference_data=self.reference_data, current_data=self.current_data
        ).save_html(path)

        lg = Logger(
            "./logs/model_monitoring.log",
            f"Column Analysis Report saved to {path}",
            "a",
        )
        lg.logging()

    def log_missing_values(self):
        reference_missing = self.reference_data.isnull().sum()
        current_missing = self.current_data.isnull().sum()

        lg = Logger(
            "./logs/model_monitoring.log",
            f"Missing values in reference data:\n{reference_missing}",
            "a",
        )
        lg.logging()

        lg = Logger(
            "./logs/model_monitoring.log",
            f"Missing values in current data:\n{current_missing}",
            "a",
        )
        lg.logging()

    def generate_model_performance_reports(self):
        y_true_train = self.predictions_train["actual"]
        y_true_test = self.predictions_test["actual"]

        for model_name in self.predictions_train.columns[1:]:
            y_pred_train = self.predictions_train[model_name]
            y_pred_test = self.predictions_test[model_name]

            performance_data_train = pd.DataFrame(
                {"target": y_true_train, "prediction": y_pred_train}
            )

            performance_data_test = pd.DataFrame(
                {"target": y_true_test, "prediction": y_pred_test}
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
            report_path = os.path.join(
                self.reports_dir, f"{model_name}_performance_report.html"
            )
            report.run(
                reference_data=dataset_train, current_data=dataset_test
            ).save_html(report_path)

            lg = Logger(
                "./logs/model_monitoring.log",
                f"{model_name} model performance report saved to {report_path}",
                "a",
            )
            lg.logging()

    def run_all(self):
        print("Running all reports...")
        self.generate_data_drift_report()
        self.generate_data_summary_report()
        self.generate_text_eval_report()
        self.generate_value_stats_report()
        self.generate_column_analysis_report()
        self.log_missing_values()
        self.generate_model_performance_reports()
        print("All reports generated successfully.")


def run_monitoring(args):
    generator = EvidentlyReportGenerator(
        reference_path=os.path.join(
            args.train_data_path, "housing_train_processed.csv"
        ),
        current_path=os.path.join(args.test_data_path, "housing_test_processed.csv"),
        pred_train_path=os.path.join(args.train_data_path, "model_predictions.csv"),
        pred_test_path=os.path.join(args.test_data_path, "model_predictions.csv"),
        reports_dir=args.report_path,
    )
    generator.run_all()
