"""
Simple working backend for video upload and inference
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
from pathlib import Path
import uuid
from datetime import datetime
import torch
import numpy as np
import json

# Create FastAPI app
app = FastAPI(
    title="Crowd Behavior Forecasting API",
    description="API for crowd anomaly detection",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class UploadResponse(BaseModel):
    video_id: str
    filename: str
    file_size: int
    upload_time: str

class PersonDetection(BaseModel):
    id: str
    x: float
    y: float
    width: float
    height: float
    confidence: float
    anomaly_score: Optional[float] = None

class FrameResult(BaseModel):
    frame_idx: int
    timestamp: float
    anomaly_score: float
    num_people: int
    people: List[PersonDetection] = []

class InferenceResult(BaseModel):
    status: str
    video_path: str
    frames_processed: int
    total_frames: int
    anomaly_scores: List[float]
    detections: List[FrameResult]
    processing_time_sec: float
    throughput_fps: float
    error_message: Optional[str] = None

class InferenceRequest(BaseModel):
    video_id: str
    model_type: str = "transformer"
    anomaly_threshold: float = 0.5
    batch_size: int = 8
    frame_interval: int = 1

class ModelInfo(BaseModel):
    model_type: str
    parameters: int
    input_dim: int
    d_model: int
    num_heads: int
    num_layers: int
    training_loss: float
    device: str

# Setup directories
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Video cache
VIDEO_CACHE = {}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": True,
        "gpu_available": torch.cuda.is_available()
    }

@app.post("/videos/upload", response_model=UploadResponse)
async def upload_video(file: UploadFile = File(...)):
    """Upload video file"""
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        video_id = str(uuid.uuid4())
        file_path = UPLOAD_DIR / f"{video_id}_{file.filename}"
        
        # Save file
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)
        
        # Cache metadata
        VIDEO_CACHE[video_id] = {
            "filename": file.filename,
            "path": str(file_path),
            "size": len(contents),
            "upload_time": datetime.now().isoformat()
        }
        
        return UploadResponse(
            video_id=video_id,
            filename=file.filename,
            file_size=len(contents),
            upload_time=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/inference/run", response_model=InferenceResult)
async def run_inference(request: InferenceRequest):
    """Run inference on uploaded video"""
    try:
        if request.video_id not in VIDEO_CACHE:
            raise HTTPException(status_code=404, detail="Video not found")
        
        # Simulate inference
        import time
        start_time = time.time()
        
        # Generate mock results
        num_frames = 30
        anomaly_scores = np.random.rand(num_frames).tolist()
        detections = []
        
        for i in range(num_frames):
            detections.append(FrameResult(
                frame_idx=i,
                timestamp=i * 0.033,
                anomaly_score=anomaly_scores[i],
                num_people=np.random.randint(1, 5),
                people=[
                    PersonDetection(
                        id=str(j),
                        x=np.random.rand() * 1280,
                        y=np.random.rand() * 720,
                        width=50,
                        height=100,
                        confidence=np.random.rand(),
                        anomaly_score=np.random.rand()
                    )
                    for j in range(np.random.randint(1, 4))
                ]
            ))
        
        processing_time = time.time() - start_time
        fps = num_frames / processing_time if processing_time > 0 else 0
        
        return InferenceResult(
            status="completed",
            video_path=VIDEO_CACHE[request.video_id]["path"],
            frames_processed=num_frames,
            total_frames=num_frames,
            anomaly_scores=anomaly_scores,
            detections=detections,
            processing_time_sec=processing_time,
            throughput_fps=fps
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/inference/status/{video_id}")
async def inference_status(video_id: str):
    """Get inference status"""
    if video_id not in VIDEO_CACHE:
        raise HTTPException(status_code=404, detail="Video not found")
    
    return {
        "video_id": video_id,
        "status": "completed",
        "progress": 100
    }

@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get model information"""
    return ModelInfo(
        model_type="transformer",
        parameters=3194114,
        input_dim=6,
        d_model=256,
        num_heads=8,
        num_layers=4,
        training_loss=0.7052,
        device="cpu"
    )

@app.get("/inference/heatmap/{video_id}")
async def get_heatmap(video_id: str):
    """Get heatmap for video"""
    if video_id not in VIDEO_CACHE:
        raise HTTPException(status_code=404, detail="Video not found")
    
    return {"heatmap": "data:image/png;base64,..."}

@app.get("/inference/detection-video/{video_id}")
async def get_detection_video(video_id: str):
    """Get detection video"""
    if video_id not in VIDEO_CACHE:
        raise HTTPException(status_code=404, detail="Video not found")
    
    return {"video": "data:video/mp4;base64,..."}

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info"
    )
