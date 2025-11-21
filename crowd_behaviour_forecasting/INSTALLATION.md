# Installation Guide

## System Requirements

### Hardware

**Minimum**
- CPU: 4 cores (Intel i5 or AMD Ryzen 5)
- RAM: 8 GB
- GPU: NVIDIA GTX 1030 (2 GB VRAM)
- Storage: 10 GB
- Network: 10 Mbps

**Recommended**
- CPU: 8 cores (Intel i7/i9 or AMD Ryzen 7/9)
- RAM: 16 GB
- GPU: NVIDIA RTX 2060 Super (8 GB VRAM) or better
- Storage: 50 GB SSD
- Network: 100 Mbps

**High-End (Production)**
- CPU: 16+ cores
- RAM: 32 GB
- GPU: NVIDIA A100/H100 (40+ GB VRAM)
- Storage: 500 GB NVMe SSD
- Network: 1 Gbps

### Software

- Ubuntu 20.04 LTS or 22.04 LTS (Recommended)
  - Alternatively: Windows 10/11 with WSL2 or CentOS 7/8
- Python 3.10 or 3.11
- NVIDIA CUDA 11.8+
- cuDNN 8.6+
- Docker 20.10+ (for containerized deployment)

## Installation Steps

### Step 1: System Preparation

#### Ubuntu/Linux
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python dependencies
sudo apt install -y python3.10 python3.10-dev python3-pip
sudo apt install -y build-essential cmake libopencv-dev python3-opencv ffmpeg

# Install NVIDIA drivers
sudo apt install -y nvidia-driver-545
sudo apt install -y nvidia-cuda-toolkit

# Verify installation
nvidia-smi
python3 --version
```

#### Windows 10/11 (with WSL2)
```powershell
# Enable WSL2
wsl --install

# Install Ubuntu 22.04 LTS
wsl --install -d Ubuntu-22.04

# In WSL2 terminal, follow Ubuntu instructions
```

#### macOS (CPU Only - No GPU Support)
```bash
# Install Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install python@3.10 opencv ffmpeg

# Install Python
python3.10 -m venv venv
source venv/bin/activate
```

### Step 2: Clone Repository

```bash
# Clone repository
git clone https://github.com/yourusername/crowd-behaviour-forecasting.git
cd crowd_behaviour_forecasting

# Or download and extract
wget https://github.com/yourusername/crowd-behaviour-forecasting/releases/download/v1.0.0/crowd-behaviour-forecasting.zip
unzip crowd-behaviour-forecasting.zip
cd crowd_behaviour_forecasting
```

### Step 3: Setup Python Environment

#### Using venv (Recommended)
```bash
# Create virtual environment
python3.10 -m venv venv

# Activate environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

#### Using Conda
```bash
# Create conda environment
conda create -n crowd-forecast python=3.10 -y
conda activate crowd-forecast

# Install mamba (faster)
conda install -c conda-forge mamba -y
```

### Step 4: Install Dependencies

#### Option A: GPU Support (NVIDIA CUDA)
```bash
# Install PyTorch with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install all dependencies
pip install -r requirements.txt

# Verify GPU availability
python -c "import torch; print(torch.cuda.is_available())"
```

#### Option B: CPU Only (No GPU)
```bash
# Install PyTorch CPU-only
pip install torch torchvision torchaudio

# Install all dependencies
pip install -r requirements.txt
```

#### Option C: Editable Development Installation
```bash
# Install in development mode
pip install -e .

# Install development tools
pip install -e ".[dev]"

# Install GPU extras
pip install -e ".[gpu]"
```

### Step 5: Download Models (Optional)

```bash
# Download pre-trained models
mkdir -p models/checkpoints

# Download GNN model
wget https://releases.example.com/models/gnn_best.pt -O models/checkpoints/gnn_best.pt

# Download Transformer model
wget https://releases.example.com/models/transformer_best.pt -O models/checkpoints/transformer_best.pt

# Download ONNX models
wget https://releases.example.com/models/gnn_quantized.onnx -O models/onnx/gnn_quantized.onnx
```

### Step 6: Verify Installation

```bash
# Test imports
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import cv2; print('OpenCV:', cv2.__version__)"
python -c "from src.data_pipeline import TrajectoryExtractor; print('Import OK')"

# List available commands
python main.py --help

# Run version check
python -c "from __init__ import __version__; print('Version:', __version__)"
```

### Step 7: Setup Datasets

```bash
# Create directory structure
python main.py setup

# Generate synthetic video for testing
python main.py setup --synthetic --duration 60

# List available datasets
python scripts/download_datasets.py --dataset list

# Download ShanghaiTech dataset (optional, ~3GB)
python scripts/download_datasets.py --dataset shanghaitech
```

## Post-Installation Configuration

### Configure for Your Hardware

Edit `configs/model_config.yaml`:

```yaml
# For GPU with 2GB VRAM
training:
  batch_size: 4
  device: "cuda"

deployment:
  quantization:
    enabled: true
    type: "int8"

# For GPU with 4+ GB VRAM
training:
  batch_size: 16
  device: "cuda"

deployment:
  quantization:
    enabled: false
    type: "fp32"

# For CPU only
training:
  batch_size: 2
  device: "cpu"
```

### Configure Environment Variables

Create `.env` file:
```bash
# CUDA settings
export CUDA_VISIBLE_DEVICES=0
export TORCH_HOME=./cache/torch_models

# API settings
export API_HOST=0.0.0.0
export API_PORT=8000

# Logging
export LOG_LEVEL=INFO
export LOG_DIR=./logs
```

Load environment:
```bash
source .env
```

## Installation Verification

### Test 1: Basic Import
```bash
python -c "from src.models.gnn_models import SpatioTemporalGCN; print('✓ GNN model OK')"
python -c "from src.models.transformer_models import TransformerBehaviorPredictor; print('✓ Transformer OK')"
python -c "from src.inference.inference_pipeline import RealtimeInferencePipeline; print('✓ Inference OK')"
```

### Test 2: Run Training
```bash
# Train on synthetic data (should complete in <1 minute)
python main.py train --model_type gnn --epochs 5 --batch_size 4
```

### Test 3: Run Inference
```bash
# Generate test video if not exists
python main.py setup --synthetic --duration 30

# Run inference
python main.py infer --video data/raw/synthetic/sample.mp4 --model models/checkpoints/gnn_final.pt
```

### Test 4: Start API Server
```bash
# Start server
python main.py server --port 8000

# In another terminal, test health
curl http://localhost:8000/health
```

### Test 5: Run Unit Tests
```bash
pytest tests/ -v
pytest tests/ --cov=src --cov-report=html
```

## Troubleshooting

### Issue: CUDA not available
```bash
# Check GPU
nvidia-smi

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### Issue: ImportError for opencv-python
```bash
# Reinstall with system libraries
pip uninstall opencv-python
sudo apt install python3-opencv
pip install opencv-python
```

### Issue: Out of Memory
```bash
# Reduce batch size in config
batch_size: 4  # from 32

# Use quantization
deployment:
  quantization:
    enabled: true
    type: "int8"
```

### Issue: YOLOv8 model download fails
```bash
# Download manually
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt
mv yolov8m.pt ~/.cache/ultralytics/

# Or use smaller model
# In code: detector = YOLOv8Detector(model_size="n")
```

### Issue: Slow on CPU
```bash
# Use int8 quantization
python main.py deploy --model best.pt --quantize

# Use ONNX Runtime
pip install onnxruntime
```

## Docker Installation (Recommended)

### Build and Run
```bash
# Build image
docker build -t crowd-forecast:latest -f deployment/docker/Dockerfile .

# Run container
docker run --gpus all \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  --name crowd-forecast \
  crowd-forecast:latest
```

### Using Docker Compose
```bash
cd deployment/docker
docker-compose up -d
docker logs -f crowd-forecast-api
docker-compose down
```

## Kubernetes Installation

```bash
# Create namespace
kubectl create namespace crowd-forecast

# Create deployment
kubectl apply -f deployment/kubernetes/deployment.yaml -n crowd-forecast

# Check status
kubectl get pods -n crowd-forecast
kubectl logs -f deployment/crowd-forecast -n crowd-forecast
```

## Uninstall

```bash
# Remove virtual environment
rm -rf venv

# Remove package installation
pip uninstall crowd-behaviour-forecasting -y

# Remove data and models
rm -rf data/ models/checkpoints/ results/

# Remove Docker images (if using Docker)
docker rmi crowd-forecast:latest
```

## Getting Help

1. **Read Documentation**: `docs/`, `README.md`, `QUICKSTART.md`
2. **Check Issues**: GitHub Issues
3. **Run Examples**: `notebooks/` directory
4. **Contact Support**: contact@example.com

## Next Steps

After successful installation:

1. **Quick Start**: Run `python main.py setup --synthetic`
2. **Train Model**: Run `python main.py train`
3. **Try Inference**: Run `python main.py infer`
4. **Start Server**: Run `python main.py server`
5. **Read Docs**: Check `docs/` directory

---

**Installation Complete!** 🎉

For questions, issues, or suggestions, please create a GitHub issue or contact the development team.
