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

## Command to deploy in ECR

Retrieve an authentication token and authenticate your Docker client to your registry. Use the AWS CLI:

```
aws ecr get-login-password --region ap-southeast-1 | docker login --username AWS --password-stdin 435141881759.dkr.ecr.ap-southeast-1.amazonaws.com
```


Build your Docker image using the following command.

```
docker build -t fsds/housing-repo .
```

After the build completes, tag your image so you can push the image to this repository:

```
docker tag fsds/housing-repo:latest 435141881759.dkr.ecr.ap-southeast-1.amazonaws.com/fsds/housing-repo:latest
```

Run the following command to push this image to your newly created AWS repository:

```
docker push 435141881759.dkr.ecr.ap-southeast-1.amazonaws.com/fsds/housing-repo:latest
```


## Command to deploy in EKS
Assuming your `kubectl` is configured for EKS cluster
# step-1: Navigate to the folder
```
cd manifests
```

# step-2: Deploy the Pod to EKS

```
kubectl apply -f housing-app-pod.yaml
```

verify deployment using:
```
kubectl get pods
```

```
kubectl describe pod housing-app
```

To see the logs:

```
kubectl logs housing-app
```

Incase you want to delete the existing Pod:

```
kubectl delete pod housing-app
```

### to tweak any configuration and any other pertinent information go to pyproject.toml