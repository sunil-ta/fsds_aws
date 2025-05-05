FROM python:3.10-slim

WORKDIR /app

# Install system packages required for building Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-venv \
    python3-dev \
    libffi-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set up virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# App settings
ENV MLFLOW_TRACKING_URI=file:///mlruns
ENV PYTHONPATH="/app:/app/src"
ENV GIT_PYTHON_REFRESH=quiet


RUN mkdir -p /mlruns
EXPOSE 5000

CMD ["python", "main.py"]
