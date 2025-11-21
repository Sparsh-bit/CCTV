# Model Architecture
This document describes the architectures of models used for crowd behavior prediction.

## 1. Spatio-Temporal GNN (Primary Model)

### Design Principle
Captures person-to-person interactions and group dynamics through message passing.

### Architecture
- **Input**: Trajectory features [x, y, vx, vy, ax, ay]
- **Layers**: Multiple GraphConv layers with ReLU activation
- **Interaction Graph**: Edges connect people within distance threshold
- **Output**: Anomaly score [0, 1] + attention weights

### Advantages
✅ Captures interaction patterns  
✅ Interpretable through attention  
✅ Efficient on edge devices  
✅ Handles variable number of agents  

### Mathematical Formulation
```
h_i^(l+1) = ReLU(W^(l) * h_i^(l) + Σ_{j∈N(i)} h_j^(l))
score_i = Sigmoid(MLP(h_i^(L)))
```

## 2. Transformer Model (Secondary)

### Design Principle
Uses self-attention to model temporal dependencies in trajectories.

### Architecture
- **Input**: Sequence of trajectory points [T, 6]
- **Positional Encoding**: Absolute position encoding
- **Transformer Blocks**: Multi-head self-attention + FFN
- **Output**: Anomaly score + attention weights

### Advantages
✅ Captures long-range dependencies  
✅ Parallel processing  
✅ Strong attention visualization  
✅ Competitive performance on benchmarks  

## 3. ConvLSTM (Tertiary)

### Design Principle
Direct video processing without explicit trajectory extraction.

### Architecture
- **Input**: Video frames [T, C, H, W]
- **ConvLSTM Cells**: Convolutional + LSTM gates
- **Output**: Anomaly map + temporal features

### Advantages
✅ End-to-end learning from video  
✅ Captures optical flow implicitly  
✅ Lower computational cost than GNN+MOT pipeline  

## Model Comparison

| Aspect | GNN | Transformer | ConvLSTM |
|--------|-----|-------------|----------|
| Speed | Fast | Medium | Medium |
| Interpretability | Excellent | Good | Fair |
| Accuracy | High | High | Medium |
| Memory | Low | Medium | Medium |
| Scalability | Excellent | Good | Fair |
| Edge-Ready | Yes | Yes | Maybe |

## Ensemble Strategy

For production deployment, we combine all three models:
- **Averaging**: Mean of prediction scores
- **Voting**: Anomaly threshold consensus
- **Weighting**: Performance-based weights from validation set

Expected improvement: +3-5% F1 score

## Training Details

### Data Requirements
- Minimum: 1000 trajectories per dataset
- Recommended: 10,000+ trajectories
- Sequence length: 30 frames (~1 second at 30fps)

### Hyperparameters (Tuned for Edge)
```yaml
GNN:
  hidden_dim: 128
  num_layers: 3
  dropout: 0.1
  
Transformer:
  d_model: 256
  num_heads: 8
  num_layers: 4
  
ConvLSTM:
  hidden_channels: 64
  num_layers: 3
```

### Loss Functions
- Primary: Binary Cross-Entropy (BCE)
- Auxiliary: Smooth L1 for regression variants

### Optimization
- Optimizer: Adam with lr=0.001
- Scheduler: Cosine annealing with warmup
- Gradient clipping: max_norm=1.0
- Early stopping: patience=10 epochs

## Performance Targets (on Edge)

| Model | Latency (ms) | Memory (MB) | F1-Score |
|-------|-------------|-----------|----------|
| GNN | <50 | 1.8 | 0.92 |
| Transformer | <80 | 2.5 | 0.90 |
| ConvLSTM | <120 | 3.2 | 0.85 |
| Ensemble | <150 | 7.5 | 0.94 |

## Edge Optimization

### ONNX Export
- Opset: 14
- Dynamic batching: Yes
- Quantization: int8 (10x smaller)

### TensorRT Acceleration
- Precision: FP32 (production), FP16 (latency-critical)
- Batch sizes: [1, 4, 8, 16]

### Memory Optimization
- Model size: ~12-42 MB (after quantization)
- Inference memory: 850 MB - 2 GB
- Suitable for edge: Yes (typical edge has 4-8GB)
