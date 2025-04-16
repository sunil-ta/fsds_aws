import subprocess
import mlflow

def run_pipeline():
    mlflow.set_experiment("Housing Price Prediction")
    with mlflow.start_run(run_name="Full_Pipeline_Run") as parent_run:
        mlflow.log_param("pipeline_stage", "start")

        # Run data ingestion
        subprocess.run(["python", "-m", "scripts.ingest_data"])
        mlflow.log_param("stage_1", "ingest_data_completed")

        # Run training
        subprocess.run(["python", "-m", "scripts.train"])
        mlflow.log_param("stage_2", "training_completed")

        # Run scoring
        subprocess.run(["python", "-m", "scripts.score"])
        mlflow.log_param("stage_3", "scoring_completed")

        # mlflow.log_param("pipeline_stage", "done")

if __name__ == "__main__":
    run_pipeline()
