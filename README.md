# MLOps Assignment 3 & 4: Observable ML with MLflow

## Overview
This project demonstrates ML experiment tracking using MLflow with a CNN model trained on CIFAR-10 dataset.

## Project Structure
- `train_mlflow.py` - Main training script with MLflow integration
- `run_experiments.py` - Script to run multiple experiments with different hyperparameters
- `mlruns/` - MLflow tracking data
- `.github/workflows/ml-pipeline.yml` - CI/CD pipeline for model validation

## Setup
```bash
pip install -r requirements.txt
```

## Running Experiments
```bash
# Start MLflow UI
mlflow ui --port 5000

# Run all experiments
python run_experiments.py
```

## View Results
Open http://localhost:5000 to view experiment results in MLflow UI.

## Author
Student ID: 202201166
