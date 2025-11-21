# Deployment Guide for Smart Cities Edge Servers

## Overview

This guide covers deploying the Crowd Behaviour Forecasting system on edge servers in smart city infrastructure (traffic control centers, CCTV hubs, etc.).

## Prerequisites

### Hardware Requirements
- **Minimum**: 
  - 4 CPU cores
  - 8 GB RAM
  - NVIDIA GPU with 2 GB VRAM (GTX 1030 or better)
  - 10 GB storage

- **Recommended**:
  - 8+ CPU cores
  - 16 GB RAM
  - NVIDIA GPU with 4+ GB VRAM (RTX 2060 or better)
  - 50 GB storage (for model variants)

### Software Requirements
- Ubuntu 20.04 LTS or 22.04 LTS
- NVIDIA CUDA 11.8+
- Docker 20.10+
- Docker Compose 1.29+

## Deployment Options

### Option 1: Docker Container (Recommended)

#### Step 1: Build Docker Image
```bash
cd deployment/docker
docker build -t crowd-forecast:latest -f Dockerfile .
```

#### Step 2: Run Container with GPU
```bash
docker run --gpus all \
  -p 8000:8000 \
  -v /data:/app/data \
  -v /models:/app/models \
  -e CUDA_VISIBLE_DEVICES=0 \
  --restart unless-stopped \
  crowd-forecast:latest
```

#### Step 3: Using Docker Compose
```bash
cd deployment/docker
docker-compose up -d
```

Monitor with:
```bash
docker logs -f crowd-forecast-api
docker stats crowd-forecast-api
```

### Option 2: Manual Installation

#### Step 1: Setup Python Environment
```bash
# Clone repository
git clone <repo-url>
cd crowd_behaviour_forecasting

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install dependencies
pip install -r requirements.txt
```

#### Step 2: Download Models
```bash
# Download pre-trained models
python scripts/download_models.py --model gnn
python scripts/download_models.py --model transformer
```

#### Step 3: Start API Server
```bash
python main.py server --host 0.0.0.0 --port 8000 --workers 4
```

### Option 3: Kubernetes Deployment

#### Step 1: Create Deployment Manifest
```yaml
# deployment/kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: crowd-forecast
spec:
  replicas: 3
  selector:
    matchLabels:
      app: crowd-forecast
  template:
    metadata:
      labels:
        app: crowd-forecast
    spec:
      containers:
      - name: crowd-forecast
        image: crowd-forecast:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "8Gi"
            cpu: "2"
            nvidia.com/gpu: "1"
          limits:
            memory: "16Gi"
            cpu: "4"
            nvidia.com/gpu: "1"
        volumeMounts:
        - name: data
          mountPath: /app/data
        - name: models
          mountPath: /app/models
      volumes:
      - name: data
        hostPath:
          path: /data/crowd-forecast
      - name: models
        hostPath:
          path: /models/crowd-forecast
```

#### Step 2: Deploy on Cluster
```bash
kubectl apply -f deployment/kubernetes/deployment.yaml
kubectl apply -f deployment/kubernetes/service.yaml
kubectl apply -f deployment/kubernetes/hpa.yaml

# Monitor
kubectl get pods -l app=crowd-forecast
kubectl logs -f deployment/crowd-forecast
```

## Configuration for Smart Cities

### Traffic Control Centers
```yaml
# Latency-critical scenario
deployment:
  batch_size: 1
  max_latency_ms: 100  # Must respond in <100ms
  quantization: int8
  tensorrt: true
  num_streams: 4  # Multiple simultaneous streams
```

### CCTV Hub Networks
```yaml
# Multi-camera scenario
inference:
  batch_size: 16  # Batch multiple cameras
  frame_buffer: 30
  detection_threshold: 0.5
  tracking_threshold: 0.7
  num_cameras: 8  # Support 8 concurrent streams
```

### Multi-Location Integration
```yaml
# Federated deployment
edge_deployment:
  central_api: true
  local_inference: true
  model_sync_interval: 3600  # Update model hourly
  enable_federation: true    # Send predictions to central
```

## Performance Tuning

### Model Selection by Hardware

**Edge Server (2GB VRAM):**
```bash
python main.py server --model gnn --quantization int8 --batch_size 4
```

**Standard Server (4GB VRAM):**
```bash
python main.py server --model ensemble --quantization fp16 --batch_size 16
```

**High-End Server (8GB+ VRAM):**
```bash
python main.py server --model ensemble --quantization fp32 --batch_size 32
```

### Batch Processing Optimization
```python
# For multiple camera feeds
batch_size = min(num_cameras, available_gpu_memory // 256)  # 256MB per stream
```

### Latency Optimization
1. **Reduce frame resolution**: 1280×720 → 640×360
2. **Skip frames**: Process every 2nd frame
3. **Use int8 quantization**: 3x speed improvement
4. **Enable TensorRT**: Additional 1.5x improvement

## API Endpoints

### Health Check
```bash
curl http://localhost:8000/health
```

### Video Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -F "video=@test.mp4"
```

### Real-time WebSocket
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/stream');
ws.onmessage = (event) => {
  const prediction = JSON.parse(event.data);
  console.log(prediction);
};
```

### Statistics
```bash
curl http://localhost:8000/stats
```

## Monitoring & Logging

### Docker Logs
```bash
docker logs --follow --timestamps crowd-forecast-api | grep -i error
```

### Prometheus Metrics
Access at: http://localhost:9090

### Performance Metrics
```bash
# Memory usage
docker stats crown-forecast-api

# GPU usage
nvidia-smi -l 1  # Refresh every 1 second

# Inference latency
curl http://localhost:8000/benchmark?num_iterations=100
```

## Scaling Strategies

### Horizontal Scaling (Multiple servers)
```bash
# Load Balancer Configuration (nginx)
upstream crowd_forecast {
    server edge1:8000;
    server edge2:8000;
    server edge3:8000;
}

server {
    location / {
        proxy_pass http://crowd_forecast;
    }
}
```

### Vertical Scaling (Single server, multiple GPUs)
```bash
docker run --gpus all \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  crowd-forecast:latest
```

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size
python main.py server --batch_size 4

# Reduce model complexity
python main.py server --model gnn  # Smaller than ensemble
```

### High Latency
```bash
# Enable quantization
python main.py deploy --model best.pt --quantize

# Enable TensorRT
python main.py deploy --model best.pt --tensorrt
```

### GPU Not Detected
```bash
# Check GPU
nvidia-smi

# Verify CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall GPU drivers
cuda-11.8-install.sh
```

## Maintenance

### Model Updates
```bash
# Download new model
wget https://releases.example.com/models/gnn_v2.pt

# Validate before deployment
python scripts/validate_model.py gnn_v2.pt

# Update in container
docker cp gnn_v2.pt crowd-forecast-api:/app/models/checkpoints/
```

### Log Rotation
```bash
# Enable log rotation
cat > /etc/logrotate.d/crowd-forecast <<EOF
/var/log/crowd-forecast/*.log {
    daily
    rotate 7
    compress
    missingok
    notifempty
}
EOF
```

### Health Checks
```bash
#!/bin/bash
while true; do
  response=$(curl -s http://localhost:8000/health)
  if [[ "$response" != *"healthy"* ]]; then
    systemctl restart crowd-forecast
    echo "Service restarted at $(date)" >> /var/log/crowd-forecast/health.log
  fi
  sleep 30
done
```

## Security

### API Authentication
```python
# Add JWT token validation
from fastapi.security import HTTPBearer

security = HTTPBearer()

@app.post("/predict")
async def predict(file: UploadFile, credentials: HTTPAuthCredentials = Depends(security)):
    # Verify token
    ...
```

### HTTPS Setup
```bash
# Generate self-signed certificate
openssl req -x509 -newkey rsa:4096 -nodes -out cert.pem -keyout key.pem -days 365

# Run with HTTPS
python main.py server --ssl-keyfile key.pem --ssl-certfile cert.pem
```

### Network Isolation
```bash
# Firewall rules
ufw allow from 192.168.1.0/24 to any port 8000
ufw deny from any to any port 8000
```

## Resources

- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)
- [Docker Documentation](https://docs.docker.com/)
- [Kubernetes GPU](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/)
- [Edge AI Best Practices](https://developer.nvidia.com/embedded-systems)
