"""
MLOps Assignment 3: Run Multiple Experiments
This script runs 5 different training configurations to compare results.

Run this after starting the MLflow UI with: mlflow ui --port 5000
"""

import subprocess
import sys

# Configuration for 5 different experiments
# Varying learning rates, batch sizes, and optimizers
EXPERIMENTS = [
    {
        "run_name": "Run1_LR_0.001_BS_64_Adam",
        "learning_rate": 0.001,
        "batch_size": 64,
        "optimizer": "Adam",
        "epochs": 10,
    },
    {
        "run_name": "Run2_LR_0.01_BS_64_Adam",
        "learning_rate": 0.01,
        "batch_size": 64,
        "optimizer": "Adam",
        "epochs": 10,
    },
    {
        "run_name": "Run3_LR_0.1_BS_64_Adam",
        "learning_rate": 0.1,  # High LR - may show instability
        "batch_size": 64,
        "optimizer": "Adam",
        "epochs": 10,
    },
    {
        "run_name": "Run4_LR_0.001_BS_128_Adam",
        "learning_rate": 0.001,
        "batch_size": 128,  # Larger batch size
        "optimizer": "Adam",
        "epochs": 10,
    },
    {
        "run_name": "Run5_LR_0.01_BS_64_SGD",
        "learning_rate": 0.01,
        "batch_size": 64,
        "optimizer": "SGD",  # Different optimizer
        "epochs": 10,
    },
]


def run_experiment(config, experiment_name, student_id):
    """Run a single experiment with the given configuration."""
    cmd = [
        sys.executable, "train_mlflow.py",
        "--experiment_name", experiment_name,
        "--student_id", student_id,
        "--run_name", config["run_name"],
        "--learning_rate", str(config["learning_rate"]),
        "--batch_size", str(config["batch_size"]),
        "--optimizer", config["optimizer"],
        "--epochs", str(config["epochs"]),
    ]

    print(f"\n{'#'*70}")
    print(f"# Starting: {config['run_name']}")
    print(f"# LR: {config['learning_rate']}, BS: {config['batch_size']}, "
          f"Opt: {config['optimizer']}, Epochs: {config['epochs']}")
    print(f"{'#'*70}\n")

    result = subprocess.run(cmd, cwd=".")
    return result.returncode == 0


def main():
    EXPERIMENT_NAME = "Assignment3_Ahmed"  
    STUDENT_ID = "202201166"  

    print("="*70)
    print("MLOps Assignment 3: Running Multiple Experiments")
    print("="*70)
    print(f"\nExperiment Name: {EXPERIMENT_NAME}")
    print(f"Student ID: {STUDENT_ID}")
    print(f"Total Runs: {len(EXPERIMENTS)}")
    print("\nMake sure MLflow UI is running: mlflow ui --port 5000")
    print("View results at: http://localhost:5000")
    print("="*70)

    successful = 0
    failed = 0

    for i, config in enumerate(EXPERIMENTS, 1):
        print(f"\n[Experiment {i}/{len(EXPERIMENTS)}]")
        if run_experiment(config, EXPERIMENT_NAME, STUDENT_ID):
            successful += 1
            print(f"Experiment {i} completed successfully!")
        else:
            failed += 1
            print(f"Experiment {i} failed!")

    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETED")
    print("="*70)
    print(f"Successful: {successful}/{len(EXPERIMENTS)}")
    print(f"Failed: {failed}/{len(EXPERIMENTS)}")
    print("\nNext Steps:")
    print("1. Open http://localhost:5000 in your browser")
    print("2. Click on your experiment to see all runs")
    print("3. Compare runs, view charts, and analyze results")
    print("4. Take screenshots for your report!")
    print("="*70)


if __name__ == "__main__":
    main()
