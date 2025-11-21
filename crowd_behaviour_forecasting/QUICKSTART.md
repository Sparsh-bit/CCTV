# Crowd Behaviour Forecasting - Quick Start Guide

## Installation (5 minutes)

### 1. Prerequisites
```bash
# Verify Python 3.10+
python --version

# Verify CUDA (if using GPU)
nvidia-smi
```

### 2. Install from Source
```bash
cd crowd_behaviour_forecasting
pip install -r requirements.txt
```

### 3. Install in Development Mode
```bash
pip install -e .
pip install -e ".[dev]"  # For development tools
```

## Quick Start (10 minutes)

### Setup
```bash
python main.py setup --synthetic --duration 60
```

### Training
```bash
python main.py train --model_type gnn --epochs 50
```

### Inference
```bash
python main.py infer \
  --video data/raw/synthetic/sample.mp4 \
  --model models/checkpoints/gnn_final.pt
```

### API Server
```bash
python main.py server --port 8000 --workers 4
```

Access at: http://localhost:8000

## What's Next

### For Research
1. Read `docs/model_details.md` - Model architecture
2. Run `scripts/benchmark_model.py` - Performance metrics
3. Modify `configs/model_config.yaml` - Hyperparameters

### For Production
1. Read `docs/deployment_guide.md` - Edge deployment
2. Build Docker: `docker build -f deployment/docker/Dockerfile .`
3. Deploy: `docker-compose up` (in deployment/docker/)

### For Datasets
```bash
python scripts/download_datasets.py --dataset shanghaitech
python scripts/download_datasets.py --dataset umn
```

## Command Reference

```bash
# List available commands
python main.py --help

# Extract trajectories
python main.py extract --video <video.mp4> --output data/processed/

# Train with Transformer
python main.py train --model_type transformer --epochs 100

# Deploy with optimization
python main.py deploy --model models/checkpoints/best.pt --quantize --tensorrt

# Benchmark
python main.py benchmark --model models/checkpoints/best.pt

# Generate synthetic video
python main.py setup --synthetic --duration 120
```

## Project Structure Overview

```
├── configs/            # Configuration files
├── data/               # Datasets (raw, processed)
├── models/             # Model checkpoints & ONNX
├── src/                # Source code
│   ├── data_pipeline/  # Video & trajectory processing
│   ├── models/         # GNN, Transformer, ConvLSTM
│   ├── inference/      # Real-time inference
│   ├── edge_deployment/# ONNX, TensorRT, API server
│   └── interpretability/ # Heatmaps, attention
├── deployment/         # Docker & Kubernetes
├── scripts/            # Utilities & benchmarks
├── tests/              # Unit tests
├── notebooks/          # Jupyter experiments
└── docs/               # Documentation
```

## Key Features

✅ **Spatio-Temporal GNN** for interaction modeling  
✅ **Transformer** for temporal patterns  
✅ **ConvLSTM** for direct video processing  
✅ **Ensemble** predictions  
✅ **Edge Optimization** (ONNX, TensorRT, int8)  
✅ **REST API** with WebSocket streaming  
✅ **Docker & Kubernetes** deployment  
✅ **Interpretability** (heatmaps, attention)  

## System Requirements

**Minimum:**
- CPU: 4 cores
- RAM: 8GB
- GPU: NVIDIA GTX 1030 (2GB)
- Storage: 10GB

**Recommended:**
- CPU: 8+ cores
- RAM: 16GB
- GPU: NVIDIA RTX 2060 (4GB+)
- Storage: 50GB

## Troubleshooting

### CUDA not detected
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Out of memory
```bash
# Reduce batch size in configs/model_config.yaml
training:
  batch_size: 16  # Reduce from 32
```

### Slow inference
```bash
# Enable quantization
python main.py deploy --model best.pt --quantize
```

## Next Steps

1. **Understand the System**: Read README.md
2. **Review Architecture**: Check docs/model_details.md
3. **Run Example**: `python main.py setup --synthetic`
4. **Train Model**: `python main.py train`
5. **Deploy**: Follow docs/deployment_guide.md
6. **Optimize**: Use `scripts/benchmark_model.py`

## Support

- **Documentation**: `docs/` folder
- **Examples**: `notebooks/` folder
- **Issues**: Create GitHub issue
- **Email**: contact@example.com

## Citation

```bibtex
@misc{crowd_behaviour_forecasting_2024,
  title={Real-Time Crowd Behaviour Forecasting for Smart Cities},
  author={Your Team},
  year={2024}
}
```

---

**Ready to get started?** Run: `python main.py setup --synthetic`
