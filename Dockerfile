# Dockerfile for ML Model Deployment
# MLOps Assignment 5

FROM python:3.10-slim

# Accept RUN_ID as a build argument
ARG RUN_ID

# Set environment variable from build arg
ENV MODEL_RUN_ID=${RUN_ID}

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Simulate downloading the model from MLflow
RUN echo "Downloading model for Run ID: ${RUN_ID}" && \
    echo "Model downloaded successfully from MLflow tracking server"

# Default command
CMD ["python", "-c", "print(f'Model container ready. Run ID: {__import__(\"os\").environ.get(\"MODEL_RUN_ID\", \"not set\")}')"]
