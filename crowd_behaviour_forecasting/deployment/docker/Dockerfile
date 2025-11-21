# Use official NVIDIA CUDA image as base
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

LABEL maintainer="Crowd Behavior Forecasting Team"
LABEL description="Real-Time Crowd Behaviour Forecasting for Smart Cities"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3-apt \
    curl \
    wget \
    git \
    build-essential \
    cmake \
    libopencv-dev \
    python3-opencv \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set Python path
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories
RUN mkdir -p data/raw data/processed models/checkpoints models/onnx results logs

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "-m", "uvicorn", "src.edge_deployment.api_server:app", "--host", "0.0.0.0", "--port", "8000"]
