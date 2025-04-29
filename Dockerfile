FROM continuumio/miniconda3

COPY . /app
WORKDIR /app

# Copy environment and project
COPY environment.yml .
RUN conda env create -f environment.yml

# Activate environment
SHELL ["conda", "run", "-n", "fsds", "/bin/bash", "-c"]

# Copy all files
COPY . .

ENV MLFLOW_TRACKING_URI=file:///mlruns
ENV PYTHONPATH="${PYTHONPATH}:/app/src"

# Create folder for mlruns
RUN mkdir -p /mlruns

# Expose MLflow UI port
EXPOSE 5000

# Default command
CMD ["conda", "run", "-n", "fsds", "python", "main.py"]