# API Reference

## Base URL
```
http://localhost:8000
```

## Authentication
Currently no authentication is required. In production, add JWT token validation.

## Endpoints

### 1. Health Check
**GET** `/health`

Check if the API is running and model is loaded.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "gpu_available": true
}
```

### 2. Predict from Video
**POST** `/predict`

Predict anomalies from uploaded video file.

**Request:**
- Form data: `video` (file) - Video file (MP4, AVI, MOV, etc.)

**Response:**
```json
{
  "anomaly_score": 0.75,
  "risk_level": "high",
  "timestamp": 0,
  "num_people": 15
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/predict \
  -F "video=@test.mp4"
```

### 3. Batch Prediction
**POST** `/predict_batch`

Predict anomalies for batch of trajectory features.

**Request Body:**
```json
[
  {
    "person_id": 1,
    "x": 100.5,
    "y": 200.3,
    "vx": 5.2,
    "vy": 3.1,
    "ax": 0.5,
    "ay": 0.2
  },
  ...
]
```

**Response:**
```json
{
  "predictions": [[0.2], [0.8], ...],
  "num_trajectories": 5
}
```

### 4. Real-time WebSocket Stream
**WS** `/ws/stream`

Stream frames in real-time and get predictions.

**Message Format (Client → Server):**
```json
{
  "type": "frame",
  "data": "<base64_encoded_frame>"
}
```

**Response (Server → Client):**
```json
{
  "anomaly_score": 0.5,
  "risk_level": "medium",
  "num_people": 8
}
```

**Example (JavaScript):**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/stream');

ws.onopen = () => {
  const canvas = document.getElementById('video');
  const ctx = canvas.getContext('2d');
  const video = document.getElementById('source');
  
  video.addEventListener('play', () => {
    const sendFrame = () => {
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const data = canvas.toDataURL('image/jpeg');
      ws.send(JSON.stringify({
        type: "frame",
        data: data.split(',')[1]  // Remove data URI prefix
      }));
      requestAnimationFrame(sendFrame);
    };
    sendFrame();
  });
};

ws.onmessage = (event) => {
  const prediction = JSON.parse(event.data);
  console.log(prediction);
};
```

### 5. List Available Models
**GET** `/models`

Get list of available model checkpoints.

**Response:**
```json
{
  "models": ["gnn_best.pt", "transformer_best.pt", "ensemble.pt"]
}
```

### 6. Load Model
**POST** `/load_model`

Load a specific model for inference.

**Query Parameters:**
- `model_name` (string, required) - Name of model file

**Response:**
```json
{
  "status": "Model loaded",
  "model": "gnn_best.pt"
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/load_model?model_name=gnn_best.pt"
```

### 7. Inference Statistics
**GET** `/stats`

Get inference statistics.

**Response:**
```json
{
  "total_inferences": 1000,
  "avg_latency_ms": 45.2,
  "median_latency_ms": 42.1,
  "min_latency_ms": 35.0,
  "max_latency_ms": 78.5,
  "throughput_fps": 22.1
}
```

### 8. Benchmark Model
**POST** `/benchmark`

Benchmark model inference latency and throughput.

**Query Parameters:**
- `num_iterations` (integer, optional, default=100) - Number of iterations
- `batch_size` (integer, optional, default=8) - Batch size

**Response:**
```json
{
  "mean_latency_ms": 45.2,
  "median_latency_ms": 42.1,
  "min_latency_ms": 35.0,
  "max_latency_ms": 78.5,
  "std_latency_ms": 8.3,
  "p95_latency_ms": 65.2,
  "p99_latency_ms": 72.5,
  "throughput_fps": 22.1
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/benchmark?num_iterations=100&batch_size=16"
```

## Error Responses

### 400 Bad Request
```json
{
  "detail": "Empty trajectory list"
}
```

### 404 Not Found
```json
{
  "detail": "Model not found: model_name.pt"
}
```

### 500 Internal Server Error
```json
{
  "detail": "Prediction failed: error message"
}
```

### 503 Service Unavailable
```json
{
  "detail": "Service not initialized"
}
```

## Rate Limiting

Currently no rate limiting is implemented. For production:
- Add rate limiting middleware
- Implement request queuing
- Set max concurrent connections

## Best Practices

1. **Batch Requests**: Send multiple trajectories in batch to improve throughput
2. **Connection Pooling**: Reuse connections for WebSocket
3. **Error Handling**: Implement retry logic with exponential backoff
4. **Monitoring**: Track latency and error rates
5. **Caching**: Cache model predictions for identical inputs

## Client Libraries

### Python
```python
import requests
import json

response = requests.post(
    'http://localhost:8000/predict',
    files={'video': open('video.mp4', 'rb')}
)
prediction = response.json()
print(prediction)
```

### JavaScript
```javascript
const response = await fetch('http://localhost:8000/predict', {
  method: 'POST',
  body: formData
});
const prediction = await response.json();
console.log(prediction);
```

### cURL
```bash
# Predict from video
curl -X POST http://localhost:8000/predict -F "video=@video.mp4"

# Check health
curl http://localhost:8000/health

# Get statistics
curl http://localhost:8000/stats
```

## Testing

Use provided test script:
```bash
python scripts/test_api.py --host localhost --port 8000
```

## Troubleshooting

### Connection refused
- Check if server is running: `curl http://localhost:8000/health`
- Check firewall rules

### Timeout
- Increase timeout value
- Reduce video size
- Check GPU memory

### Model not found
- Verify model path: `curl http://localhost:8000/models`
- Download model: `python scripts/download_models.py`

## Performance Tips

1. Use smaller batch sizes for low-latency requirements
2. Enable GPU acceleration with CUDA
3. Use int8 quantization for faster inference
4. Preload model: `/load_model` before predictions
5. Monitor GPU memory with `nvidia-smi`

## Support

For issues or questions:
- GitHub Issues: https://github.com/yourusername/crowd-behaviour-forecasting/issues
- Documentation: https://crowd-behaviour-forecasting.readthedocs.io/
- Email: contact@example.com
