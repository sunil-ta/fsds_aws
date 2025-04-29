# Median housing value prediction

The housing data can be downloaded from https://raw.githubusercontent.com/ageron/handson-ml/master/. The script has codes to download the data. We have modelled the median house value on given housing data.

The following techniques have been used:

 - Linear regression
 - Decision Tree
 - Random Forest
 - Random Search CV
 - Grid Search CV

## Steps performed
 - We prepare and clean the data. We check and impute for missing values.
 - Features are generated and the variables are checked for correlation.
 - Multiple sampling techinuqies are evaluated. The data set is split into train and test.
 - All the above said modelling techniques are tried and evaluated. The final metric used to evaluate is mean squared error.

## Download the .whl or tar.gz file for the package.
Open your terminal and navigate to the directory where the file is located.


## Command to activate the environment

```
conda activate fsds
```

## To excute the script

``` python
python main.py
```

## Command to run the ML flow UI to port 5000

```
mlflow server --backend-store-uri mlruns/ --default-artifact-root mlruns/ --host 0.0.0.0 --port 5000

```

## Open port 5000 to track ml flow experiments

```
http://localhost:5000/

```
## to look for logs - go to logs folder.

# docker setup

## 1. Build Docker Image
Make sure you are in the root directory (where the Dockerfile is located), and run:
```
docker build -t fsds-model .
```

## 2. Run Docker Container
To simply run the training pipeline:
```
docker run fsds-model
```

To expose MLflow UI on port 5000:
```
docker run -p 5000:5000 -v $(pwd)/mlruns:/mlruns fsds-model
```

Then in other terminal run below one:
```
mlflow ui --backend-store-uri ./mlruns --port 5000
```

or click below link to track the expirement in browser

```
http://localhost:5000
```

### to tweak any configuration and any other pertinent information go to pyproject.toml