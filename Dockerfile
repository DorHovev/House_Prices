# Use Python base image
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Create necessary directories
RUN mkdir -p data saved_models logs

# Set environment variables
ENV PYTHONPATH=/app
ENV DATA_PATH=/app/data
ENV MODELS_PATH=/app/saved_models
ENV LOGS_PATH=/app/logs

# Set default command
CMD ["python", "src/train.py"]