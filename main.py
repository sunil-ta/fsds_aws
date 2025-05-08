import subprocess

import mlflow


def run_pipeline():
    mlflow.set_experiment("Housing Price Prediction")
    with mlflow.start_run(run_name="Full_Pipeline_Run") as parent_run:
        mlflow.log_param("pipeline_stage", "start")

        # Run data ingestion
        with mlflow.start_run(
            run_name="housing_data_ingestion", nested=True
        ) as child_run:
            subprocess.run(["python", "-m", "scripts.ingest_data"])
            mlflow.log_param("stage_1", "ingest_data_completed")

        # Run training
        with mlflow.start_run(run_name="model_training", nested=True) as child_run:
            subprocess.run(["python", "-m", "scripts.train", child_run.info.run_id])
            mlflow.log_param("stage_2", "training_completed")

        # Run scoring
        with mlflow.start_run(run_name="model scoring", nested=True) as child_run:
            subprocess.run(["python", "-m", "scripts.score", child_run.info.run_id])
            mlflow.log_param("stage_3", "scoring_completed")

        # mlflow.log_param("pipeline_stage", "done")


if __name__ == "__main__":
    # run_pipeline()
    subprocess.run(["python", "-m", "scripts.model_monitoring"])
