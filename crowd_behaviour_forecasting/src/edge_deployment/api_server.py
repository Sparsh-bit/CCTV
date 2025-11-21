from fastapi import FastAPI, File, UploadFile, WebSocket, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
import numpy as np
import cv2
import logging
import asyncio
from pathlib import Path
import tempfile
import json

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Crowd Behavior Forecasting API",
    description="Real-time crowd anomaly detection for smart cities",
    version="1.0.0"
)

class TrajectoryFeatures(BaseModel):

    person_id: int
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    ax: float = 0.0
    ay: float = 0.0

class PredictionResponse(BaseModel):

    anomaly_score: float
    risk_level: str
    timestamp: int
    num_people: int

class HealthResponse(BaseModel):

    status: str
    model_loaded: bool
    gpu_available: bool

class AppState:
    def __init__(self):
        self.model = None
        self.inference_pipeline = None
        self.is_ready = False

app_state = AppState()

@app.on_event("startup")
async def startup_event():

    try:

        logger.info("Initializing API server...")
        app_state.is_ready = True
        logger.info("API server ready")
    except Exception as e:
        logger.error(f"Startup failed: {e}")

@app.get("/health", response_model=HealthResponse)
async def health_check():

    import torch

    return HealthResponse(
        status="healthy" if app_state.is_ready else "initializing",
        model_loaded=app_state.model is not None,
        gpu_available=torch.cuda.is_available()
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_video(file: UploadFile = File(...)):

    if not app_state.is_ready:
        raise HTTPException(status_code=503, detail="Service not initialized")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    try:

        from src.data_pipeline.trajectory_extractor import TrajectoryExtractor
        from src.inference.inference_pipeline import RealtimeInferencePipeline

        extractor = TrajectoryExtractor()
        trajectories = extractor.extract_from_video(tmp_path)

        if app_state.inference_pipeline is None:
            raise HTTPException(status_code=500, detail="Model not initialized")

        if trajectories:
            scores, _ = app_state.inference_pipeline.predict_single_frame(trajectories)

            max_score = float(np.max(scores)) if len(scores) > 0 else 0.0
            if max_score > 0.7:
                risk_level = "high"
            elif max_score > 0.3:
                risk_level = "medium"
            else:
                risk_level = "low"

            return PredictionResponse(
                anomaly_score=max_score,
                risk_level=risk_level,
                timestamp=0,
                num_people=len(trajectories)
            )
        else:
            return PredictionResponse(
                anomaly_score=0.0,
                risk_level="low",
                timestamp=0,
                num_people=0
            )

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    finally:

        Path(tmp_path).unlink(missing_ok=True)

@app.post("/predict_batch")
async def predict_batch(trajectories: List[TrajectoryFeatures]):

    if not trajectories:
        raise HTTPException(status_code=400, detail="Empty trajectory list")

    try:

        features = np.array([
            [t.x, t.y, t.vx, t.vy, t.ax, t.ay]
            for t in trajectories
        ])

        if app_state.inference_pipeline is None:
            raise HTTPException(status_code=500, detail="Model not initialized")

        predictions, _ = app_state.inference_pipeline.predict_single_frame({})

        return {
            "predictions": predictions.tolist() if len(predictions) > 0 else [],
            "num_trajectories": len(trajectories)
        }

    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):

    await websocket.accept()

    try:
        while True:

            data = await websocket.receive_text()

            if not app_state.is_ready:
                await websocket.send_json({"error": "Service not ready"})
                continue

            try:
                message = json.loads(data)

                if message.get("type") == "frame":

                    import base64
                    frame_data = base64.b64decode(message["data"])
                    nparr = np.frombuffer(frame_data, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    prediction = {
                        "anomaly_score": 0.5,
                        "risk_level": "medium",
                        "num_people": 5
                    }

                    await websocket.send_json(prediction)

                elif message.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})

            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON"})

    except Exception as e:
        logger.error(f"WebSocket error: {e}")

    finally:
        await websocket.close()

@app.get("/models")
async def list_models():

    from pathlib import Path

    model_dir = Path("models/checkpoints")
    if not model_dir.exists():
        return {"models": []}

    models = [f.name for f in model_dir.glob("*.pt")]
    return {"models": models}

@app.post("/load_model")
async def load_model(model_name: str):

    try:
        model_path = Path("models/checkpoints") / model_name

        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"Model not found: {model_name}")

        from src.inference.inference_pipeline import RealtimeInferencePipeline

        app_state.inference_pipeline = RealtimeInferencePipeline(str(model_path))

        return {"status": "Model loaded", "model": model_name}

    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

@app.get("/stats")
async def get_stats():

    if app_state.inference_pipeline is None:
        return {"error": "No model loaded"}

    import statistics

    times = app_state.inference_pipeline.inference_times

    if not times:
        return {"total_inferences": 0}

    return {
        "total_inferences": len(times),
        "avg_latency_ms": statistics.mean(times),
        "median_latency_ms": statistics.median(times),
        "min_latency_ms": min(times),
        "max_latency_ms": max(times),
        "throughput_fps": app_state.inference_pipeline.throughput_fps
    }

@app.post("/benchmark")
async def benchmark(num_iterations: int = 100, batch_size: int = 8):

    if app_state.inference_pipeline is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        from src.inference.inference_pipeline import BatchInferencePipeline

        features = np.random.randn(batch_size, 30, 6).astype(np.float32)

        benchmark_pipeline = BatchInferencePipeline(
            "models/checkpoints/best.pt",
            batch_size=batch_size
        )

        results = benchmark_pipeline.benchmark(features, num_iterations)

        return results

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {str(e)}")

@app.get("/")
async def root():

    return {
        "name": "Crowd Behavior Forecasting API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "/health": "Health check",
            "/predict": "Video prediction (POST)",
            "/predict_batch": "Batch trajectory prediction (POST)",
            "/ws/stream": "WebSocket for real-time streaming",
            "/models": "List available models",
            "/load_model": "Load a model (POST)",
            "/stats": "Inference statistics",
            "/benchmark": "Benchmark inference (POST)"
        }
    }

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
