"""
MLOps Assignment 5: Accuracy Threshold Check Script
Reads the Run ID from model_info.txt, checks accuracy in MLflow,
and fails the pipeline if accuracy is below 0.85.
"""

import os
import sys
import mlflow

# Threshold for model deployment
ACCURACY_THRESHOLD = 0.85

# Get MLflow tracking URI from environment variable
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def main():
    # Read Run ID from model_info.txt
    try:
        with open("model_info.txt", "r") as f:
            run_id = f.read().strip()
    except FileNotFoundError:
        print("ERROR: model_info.txt not found!")
        sys.exit(1)

    if not run_id:
        print("ERROR: Run ID is empty!")
        sys.exit(1)

    print(f"Checking accuracy for Run ID: {run_id}")
    print(f"MLflow Tracking URI: {MLFLOW_TRACKING_URI}")

    # Get the run from MLflow
    try:
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
    except Exception as e:
        print(f"ERROR: Could not retrieve run from MLflow: {e}")
        sys.exit(1)

    # Get the accuracy metric
    metrics = run.data.metrics
    accuracy = metrics.get("accuracy")

    if accuracy is None:
        # Try best_test_accuracy (stored as percentage) and convert
        best_test_accuracy = metrics.get("best_test_accuracy")
        if best_test_accuracy is not None:
            accuracy = best_test_accuracy / 100.0
        else:
            print("ERROR: No accuracy metric found in the run!")
            sys.exit(1)

    print(f"Model Accuracy: {accuracy:.4f}")
    print(f"Threshold: {ACCURACY_THRESHOLD}")

    # Check threshold
    if accuracy < ACCURACY_THRESHOLD:
        print(f"FAILED: Accuracy {accuracy:.4f} is below threshold {ACCURACY_THRESHOLD}")
        print("Deployment blocked - model does not meet quality requirements.")
        sys.exit(1)
    else:
        print(f"PASSED: Accuracy {accuracy:.4f} meets threshold {ACCURACY_THRESHOLD}")
        print("Model approved for deployment!")
        sys.exit(0)


if __name__ == "__main__":
    main()
