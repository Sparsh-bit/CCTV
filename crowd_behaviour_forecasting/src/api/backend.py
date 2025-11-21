"""
FastAPI Backend Integration Guide for Crowd Behavior Forecasting System
This module provides all necessary APIs for frontend integration
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import cv2
import torch
from pathlib import Path
import uuid
import asyncio
import json
from datetime import datetime

from src.inference.inference_pipeline import RealtimeInferencePipeline
from src.models.transformer_models import TransformerBehaviorPredictor
from src.data_pipeline.video_loader import VideoLoader

# Pydantic Models
class InferenceRequest(BaseModel):
    video_id: str
    model_type: str = "transformer"
    anomaly_threshold: float = 0.5
    batch_size: int = 8
    frame_interval: int = 1

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

class ModelInfo(BaseModel):
    model_type: str
    parameters: int
    input_dim: int
    d_model: int
    num_heads: int
    num_layers: int
    training_loss: float
    device: str

# Initialize FastAPI app
app = FastAPI(
    title="Crowd Behavior Forecasting API",
    description="API for crowd anomaly detection and trajectory analysis",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Model cache
MODEL_CACHE = {}
VIDEO_CACHE = {}

def get_model(model_type: str = "transformer") -> torch.nn.Module:
    """Get or create model from cache"""
    if model_type not in MODEL_CACHE:
        device = torch.device("cpu")
        if model_type == "transformer":
            model = TransformerBehaviorPredictor(
                input_dim=6,
                d_model=256,
                num_heads=8,
                num_layers=4
            )
            checkpoint = torch.load(
                "models/checkpoints/transformer_final.pt",
                map_location=device
            )
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            model.to(device)
            model.eval()
            MODEL_CACHE[model_type] = model
    return MODEL_CACHE[model_type]

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/videos/upload", response_model=UploadResponse, tags=["Videos"])
async def upload_video(file: UploadFile = File(...)):
    """Upload video file"""
    try:
        video_id = str(uuid.uuid4())
        file_path = UPLOAD_DIR / f"{video_id}_{file.filename}"
        
        # Save file
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)
        
        # Cache video metadata
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
            upload_time=VIDEO_CACHE[video_id]["upload_time"]
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/inference/run", response_model=InferenceResult, tags=["Inference"])
async def run_inference(request: InferenceRequest, background_tasks: BackgroundTasks):
    """Run inference on uploaded video"""
    try:
        if request.video_id not in VIDEO_CACHE:
            raise HTTPException(status_code=404, detail="Video not found")
        
        video_path = VIDEO_CACHE[request.video_id]["path"]
        # Kick off background processing using the real pipeline
        def _run_pipeline(video_id: str, video_path: str, req: InferenceRequest):
            import time
            try:
                pipeline = RealtimeInferencePipeline(
                    model_path="models/checkpoints/transformer_final.pt",
                    device="cpu",
                    batch_size=req.batch_size,
                    frame_buffer_size=30
                )

                # Process whole video (this may be slow on CPU)
                results = pipeline.process_video(video_path, skip_frames=req.frame_interval)

                # Build output structures
                anomaly_scores = []
                detections_out = []

                # Determine video metadata
                loader = VideoLoader()
                width, height, fps, frame_count = loader.load_video(video_path)

                # Iterate frames and trajectories
                for idx, (scores, trajs) in enumerate(zip(results.get('anomaly_scores', []), results.get('trajectories', []))):
                    # scores: numpy array per-trajectory
                    people = []
                    # For each trajectory, take last point as current location
                    for t_idx, (person_id, traj) in enumerate(trajs.items()):
                        if len(traj.points) == 0:
                            continue
                        last = traj.points[-1]

                        # Use last detection bbox if available
                        if hasattr(last, 'x1') and last.x2 and last.x2 > last.x1:
                            x1 = int(max(0, min(width - 1, last.x1)))
                            y1 = int(max(0, min(height - 1, last.y1)))
                            x2 = int(max(0, min(width - 1, last.x2)))
                            y2 = int(max(0, min(height - 1, last.y2)))
                            bw = max(1, x2 - x1)
                            bh = max(1, y2 - y1)
                            x = x1
                            y = y1
                        else:
                            # Approximate bbox size
                            bw, bh = 60, 150
                            x = max(0, min(width - bw, int(last.x - bw // 2)))
                            y = max(0, min(height - bh, int(last.y - bh // 2)))

                        score_val = float(scores[t_idx]) if hasattr(scores, '__len__') and len(scores) > t_idx else float(np.mean(scores) if len(scores) else 0.0)

                        people.append(PersonDetection(
                            id=str(person_id),
                            x=float(x),
                            y=float(y),
                            width=float(bw),
                            height=float(bh),
                            confidence=1.0,
                            anomaly_score=score_val
                        ))

                    avg_anom = float(np.mean([p.anomaly_score for p in people])) if people else 0.0
                    anomaly_scores.append(avg_anom)

                    detections_out.append(FrameResult(
                        frame_idx=results.get('timestamps', [idx])[idx] if 'timestamps' in results and len(results['timestamps'])>idx else idx,
                        timestamp=(results.get('timestamps', [idx])[idx] / fps) if fps>0 else 0.0,
                        anomaly_score=avg_anom,
                        num_people=len(people),
                        people=people
                    ))

                processing_time = sum(pipeline.inference_times) / 1000.0 if pipeline.inference_times else 0.0
                throughput = pipeline.throughput_fps if hasattr(pipeline, 'throughput_fps') else 0.0

                result = InferenceResult(
                    status="completed",
                    video_path=video_path,
                    frames_processed=len(detections_out),
                    total_frames=frame_count,
                    anomaly_scores=anomaly_scores,
                    detections=detections_out,
                    processing_time_sec=processing_time,
                    throughput_fps=throughput
                )

                # Save results JSON
                result_path = RESULTS_DIR / f"{video_id}_result.json"
                with open(result_path, "w") as f:
                    json.dump(result.dict(), f, indent=2)

                # Create heatmap by accumulating anomalies on a grid
                heatmap = np.zeros((720, 1280), dtype=np.float32)
                for frame_idx, frame_trajs in enumerate(results.get('trajectories', [])):
                    scores = results.get('anomaly_scores', [])[frame_idx] if frame_idx < len(results.get('anomaly_scores', [])) else []
                    for t_idx, (person_id, traj) in enumerate(frame_trajs.items()):
                        if len(traj.points) == 0:
                            continue
                        last = traj.points[-1]
                        x = int(max(0, min(1279, last.x)))
                        y = int(max(0, min(719, last.y)))
                        s = float(scores[t_idx]) if hasattr(scores, '__len__') and len(scores) > t_idx else 0.0
                        heatmap[y, x] += s

                # Normalize heatmap and write color map
                if heatmap.max() > 0:
                    hm = (heatmap / heatmap.max() * 255).astype(np.uint8)
                else:
                    hm = (heatmap * 255).astype(np.uint8)
                hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
                heatmap_path = RESULTS_DIR / f"{video_id}_heatmap.png"
                cv2.imwrite(str(heatmap_path), hm_color)

                # Create overlay video (draw detections on frames)
                overlay_path = RESULTS_DIR / f"{video_id}_overlay.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(str(overlay_path), fourcc, fps if fps>0 else 10, (width, height))
                for f_idx, frame in enumerate(results.get('frames', [])):
                    vis = frame.copy()
                    frame_trajs = results.get('trajectories', [])[f_idx]
                    scores = results.get('anomaly_scores', [])[f_idx] if f_idx < len(results.get('anomaly_scores', [])) else []
                    for t_idx, (person_id, traj) in enumerate(frame_trajs.items()):
                        if len(traj.points) == 0:
                            continue
                        last = traj.points[-1]
                        # Use bbox if available
                        if hasattr(last, 'x1') and last.x2 and last.x2 > last.x1:
                            x1 = int(max(0, min(width - 1, last.x1)))
                            y1 = int(max(0, min(height - 1, last.y1)))
                            x2 = int(max(0, min(width - 1, last.x2)))
                            y2 = int(max(0, min(height - 1, last.y2)))
                        else:
                            bw, bh = 60, 150
                            x1 = int(max(0, min(width - bw, last.x - bw // 2)))
                            y1 = int(max(0, min(height - bh, last.y - bh // 2)))
                            x2 = x1 + bw
                            y2 = y1 + bh

                        score_val = float(scores[t_idx]) if hasattr(scores, '__len__') and len(scores) > t_idx else 0.0
                        color = (0, int(min(255, score_val * 255)), int(max(0, 255 - score_val * 255)))
                        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(vis, f"ID:{person_id} A:{score_val:.2f}", (x1, max(0, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    out.write(vis)
                out.release()

            except Exception as e:
                # Save error result
                err_result = InferenceResult(
                    status="error",
                    video_path=video_path,
                    frames_processed=0,
                    total_frames=0,
                    anomaly_scores=[],
                    detections=[],
                    processing_time_sec=0.0,
                    throughput_fps=0.0,
                    error_message=str(e)
                )
                result_path = RESULTS_DIR / f"{video_id}_result.json"
                with open(result_path, "w") as f:
                    json.dump(err_result.dict(), f, indent=2)

        # schedule background task
        background_tasks.add_task(_run_pipeline, request.video_id, video_path, request)

        # Return immediate processing response
        return InferenceResult(
            status="processing",
            video_path=video_path,
            frames_processed=0,
            total_frames=0,
            anomaly_scores=[],
            detections=[],
            processing_time_sec=0.0,
            throughput_fps=0.0
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/inference/status/{video_id}", response_model=InferenceResult, tags=["Inference"])
async def get_inference_status(video_id: str):
    """Get inference status"""
    try:
        result_path = RESULTS_DIR / f"{video_id}_result.json"
        if result_path.exists():
            with open(result_path, "r") as f:
                result_data = json.load(f)
            return InferenceResult(**result_data)
        else:
            raise HTTPException(status_code=404, detail="Results not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
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

@app.get("/inference/heatmap/{video_id}", tags=["Results"])
async def get_heatmap(video_id: str):
    """Get anomaly heatmap"""
    try:
        # If a generated heatmap exists from pipeline, return it; otherwise generate demo heatmap
        heatmap_path = RESULTS_DIR / f"{video_id}_heatmap.png"
        if heatmap_path.exists():
            return FileResponse(heatmap_path, media_type="image/png")

        heatmap = np.random.rand(720, 1280)
        heatmap = (heatmap * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Save and return
        cv2.imwrite(str(heatmap_path), heatmap_color)
        return FileResponse(heatmap_path, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/inference/detection-video/{video_id}", tags=["Results"])
async def get_detection_video(video_id: str):
    """Get video with detection overlays"""
    try:
        # Prefer overlay video if available
        overlay_path = RESULTS_DIR / f"{video_id}_overlay.mp4"
        if overlay_path.exists():
            return FileResponse(overlay_path, media_type="video/mp4")

        if video_id not in VIDEO_CACHE:
            raise HTTPException(status_code=404, detail="Video not found")

        video_path = VIDEO_CACHE[video_id]["path"]
        return FileResponse(video_path, media_type="video/mp4")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
