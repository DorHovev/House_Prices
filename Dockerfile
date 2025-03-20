FROM tensorflow/tensorflow:2.13.0

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY src/ ./src/
COPY models/ ./models/
COPY .env .

# Create necessary directories
RUN mkdir -p data saved_models logs

# Set environment variables
ENV PYTHONPATH=/app

# Command to run training
CMD ["python", "src/train.py"]