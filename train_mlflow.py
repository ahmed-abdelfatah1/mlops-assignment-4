"""
MLOps Assignment 3: Observable ML with MLflow
PyTorch Training Script with MLflow Tracking (GPU Optimized)

This script trains a CNN on CIFAR-10 dataset and logs all metrics,
parameters, and artifacts to MLflow for experiment tracking.
"""

import argparse
import os
import mlflow
import mlflow.pytorch
import torch

# Get MLflow tracking URI from environment variable (GitHub secret) or use local SQLite
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time


def setup_device():
    """Setup and configure the computing device (GPU/CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # Enable cuDNN auto-tuner to find the best algorithm for your hardware
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

        # Print GPU info
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Detected: {gpu_name}")
        print(f"GPU Memory: {gpu_memory:.2f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Enabled: {torch.backends.cudnn.enabled}")
    else:
        device = torch.device("cpu")
        print("No GPU detected, using CPU")

    return device


# Define a simple CNN model for CIFAR-10
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # 32x32 -> 16x16
        x = self.pool(self.relu(self.conv2(x)))  # 16x16 -> 8x8
        x = self.pool(self.relu(self.conv3(x)))  # 8x8 -> 4x4
        x = x.view(-1, 64 * 4 * 4)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


def get_data_loaders(batch_size, use_cuda=False):
    """Create train and test data loaders for CIFAR-10 with GPU optimization."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )

    # GPU optimization: pin_memory speeds up CPU to GPU transfer
    # num_workers=0 for Windows compatibility (avoids multiprocessing issues)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=use_cuda  # Faster data transfer to GPU
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=use_cuda
    )

    return train_loader, test_loader


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch and return average loss and accuracy."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        # non_blocking=True allows async CPU to GPU transfer when using pin_memory
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def evaluate(model, test_loader, criterion, device):
    """Evaluate the model on test data."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            # non_blocking=True allows async CPU to GPU transfer
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(test_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def main(args):
    # Set device with GPU optimizations
    device = setup_device()
    use_cuda = device.type == "cuda"
    print(f"Using device: {device}")

    # Set the MLflow experiment name
    mlflow.set_experiment(args.experiment_name)

    # Start MLflow run
    with mlflow.start_run(run_name=args.run_name):
        # Log tags
        mlflow.set_tag("student_id", args.student_id)
        mlflow.set_tag("model_type", "SimpleCNN")
        mlflow.set_tag("dataset", "CIFAR-10")
        mlflow.set_tag("framework", "PyTorch")

        # Log hyperparameters
        mlflow.log_param("learning_rate", args.learning_rate)
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("optimizer", args.optimizer)
        mlflow.log_param("device", str(device))

        print(f"\n{'='*60}")
        print(f"MLflow Run Started")
        print(f"Experiment: {args.experiment_name}")
        print(f"Run Name: {args.run_name}")
        print(f"{'='*60}")
        print(f"Hyperparameters:")
        print(f"  Learning Rate: {args.learning_rate}")
        print(f"  Epochs: {args.epochs}")
        print(f"  Batch Size: {args.batch_size}")
        print(f"  Optimizer: {args.optimizer}")
        print(f"{'='*60}\n")

        # Log GPU info if available
        if use_cuda:
            mlflow.log_param("gpu_name", torch.cuda.get_device_name(0))
            mlflow.log_param("cuda_version", torch.version.cuda)

        # Get data loaders with GPU optimization
        train_loader, test_loader = get_data_loaders(args.batch_size, use_cuda=use_cuda)

        # Initialize model, criterion, and optimizer
        model = SimpleCNN().to(device)
        criterion = nn.CrossEntropyLoss()

        if args.optimizer == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        elif args.optimizer == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
        else:
            optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate)

        # Training loop
        best_test_accuracy = 0.0
        start_time = time.time()

        for epoch in range(1, args.epochs + 1):
            # Train
            train_loss, train_accuracy = train_epoch(
                model, train_loader, criterion, optimizer, device
            )

            # Evaluate
            test_loss, test_accuracy = evaluate(
                model, test_loader, criterion, device
            )

            # Log metrics for this epoch (Live Logging)
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_accuracy, step=epoch)
            mlflow.log_metric("test_loss", test_loss, step=epoch)
            mlflow.log_metric("test_accuracy", test_accuracy, step=epoch)

            # Track best accuracy
            if test_accuracy > best_test_accuracy:
                best_test_accuracy = test_accuracy

            print(f"Epoch [{epoch}/{args.epochs}] "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}% | "
                  f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%")

        training_time = time.time() - start_time

        # Log final metrics
        mlflow.log_metric("best_test_accuracy", best_test_accuracy)
        mlflow.log_metric("accuracy", best_test_accuracy / 100.0)  # Decimal for threshold check
        mlflow.log_metric("final_train_loss", train_loss)
        mlflow.log_metric("final_test_loss", test_loss)
        mlflow.log_metric("training_time_seconds", training_time)

        # Log the model using MLflow PyTorch flavor
        print("\nSaving model to MLflow...")
        mlflow.pytorch.log_model(
            model,
            "model",
            registered_model_name=None,  # Set a name if you want to register it
        )

        # Get the Run ID
        run_id = mlflow.active_run().info.run_id

        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Best Test Accuracy: {best_test_accuracy:.2f}%")
        print(f"Training Time: {training_time:.2f} seconds")
        print(f"{'='*60}")
        print(f"\nView results at: http://localhost:5000")
        print(f"Run ID: {run_id}")

        # Export Run ID to model_info.txt for CI/CD pipeline
        with open("model_info.txt", "w") as f:
            f.write(run_id)
        print(f"Run ID exported to model_info.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch CIFAR-10 Training with MLflow")

    parser.add_argument("--experiment_name", type=str,
                        default="Assignment3_Ahmed",
                        help="MLflow experiment name")
    parser.add_argument("--run_name", type=str,
                        default=None,
                        help="Name for this specific run")
    parser.add_argument("--student_id", type=str,
                        default="YOUR_ID",
                        help="Your student ID for tagging")
    parser.add_argument("--learning_rate", type=float,
                        default=0.001,
                        help="Learning rate")
    parser.add_argument("--epochs", type=int,
                        default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int,
                        default=64,
                        help="Batch size for training")
    parser.add_argument("--optimizer", type=str,
                        default="Adam",
                        choices=["Adam", "SGD", "RMSprop"],
                        help="Optimizer to use")

    args = parser.parse_args()
    main(args)
