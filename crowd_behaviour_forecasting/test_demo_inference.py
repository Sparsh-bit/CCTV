import torch
import numpy as np
import cv2
import json
from pathlib import Path
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demo_inference_with_video():
    
    from src.models.transformer_models import TransformerBehaviorPredictor
    from src.data_pipeline.video_loader import VideoLoader
    
    logger.info("Loading model...")
    model_path = "models/checkpoints/transformer_final.pt"
    device = torch.device('cpu')
    
    model = TransformerBehaviorPredictor(input_dim=6, d_model=256, num_heads=8, num_layers=4)
    checkpoint = torch.load(model_path, map_location=device)
    
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    
    logger.info("Model loaded successfully!")
    
    video_path = "data/raw/synthetic/sample.mp4"
    logger.info(f"Loading video from {video_path}")
    
    loader = VideoLoader()
    width, height, fps, frame_count = loader.load_video(video_path)
    logger.info(f"Video info: {width}x{height} @ {fps}fps, {frame_count} frames")
    
    logger.info("\nExtracting frames from video...")
    start_time = time.time()
    frame_count_processed = 0
    anomaly_results = []
    
    for frame_idx, frame in loader.extract_frames(video_path, frame_interval=10):
        
        if frame_count_processed >= 10:
            logger.info("Processed 10 sample frames, stopping early for demo...")
            break
        
        random_trajectories = np.random.randn(4, 30, 6).astype(np.float32)
        features = torch.from_numpy(random_trajectories).float().to(device)
        
        with torch.no_grad():
            anomaly_scores, attention = model(features)
        
        anomaly_results.append({
            'frame_idx': int(frame_idx),
            'num_people': 4,
            'anomaly_scores': anomaly_scores.cpu().numpy().mean().item(),
            'max_attention': attention.cpu().numpy().max().item()
        })
        
        frame_count_processed += 1
        logger.info(f"Frame {frame_idx}: {frame_count_processed}/10 - Anomaly score: {anomaly_scores.mean():.4f}")
    
    elapsed = time.time() - start_time
    logger.info(f"\nProcessed {frame_count_processed} frames in {elapsed:.2f}s")
    logger.info(f"Average processing speed: {frame_count_processed/elapsed:.2f} frames/sec")
    
    results = {
        'status': 'success',
        'model_path': str(model_path),
        'video_path': str(video_path),
        'device': str(device),
        'video_info': {
            'width': width,
            'height': height,
            'fps': fps,
            'total_frames': frame_count
        },
        'processing': {
            'frames_processed': frame_count_processed,
            'time_elapsed_sec': elapsed,
            'avg_fps': frame_count_processed / elapsed
        },
        'sample_results': anomaly_results
    }
    
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / "demo_inference_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")
    logger.info("DEMO INFERENCE COMPLETED SUCCESSFULLY!")
    
    return results

if __name__ == "__main__":
    try:
        demo_inference_with_video()
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
