# Median housing value prediction

The housing data can be downloaded from https://raw.githubusercontent.com/ageron/handson-ml/master/. The script has codes to download the data. We have modelled the median house value on given housing data.

The following techniques have been used:

 - Linear regression
 - Decision Tree
 - Random Forest

## Steps performed
 - We prepare and clean the data. We check and impute for missing values.
 - Features are generated and the variables are checked for correlation.
 - Multiple sampling techinuqies are evaluated. The data set is split into train and test.
 - All the above said modelling techniques are tried and evaluated. The final metric used to evaluate is mean squared error.

## Download the .whl or tar.gz file for the package.
Open your terminal and navigate to the directory where the file is located.

```
pip install package_name.whl
```
```
tar -xvzf package_name.tar.gz
```
Replace package_name with the name of the package you want to install.
```
cd package_name
```
## Build docker image
```
docker build -t mle-training
```
### run docker container
```
docker run -it mle-training
```

## Command to create an environment
```
conda env create -f environment/env.yml
```

## Command to activate the environment

```
conda activate mle-dev
```

## To excute the script

``` python
python nonstandardcode.py
```

## Command to run the ML flow UI to port 5001

```
mlflow server --backend-store-uri mlruns/ --default-artifact-root mlruns/ --host 0.0.0.0 --port 5001

```

## Open port 5001 to track ml flow experiments

```
[http://localhost:5001/#/

```
## to look for logs - go to logs folder.

### to tweak any configuration and any other pertinent information go to pyproject.toml