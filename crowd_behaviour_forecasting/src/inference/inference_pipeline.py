import torch
import numpy as np
import cv2
from typing import Dict, Tuple, Optional, List
from pathlib import Path
import logging
import time

logger = logging.getLogger(__name__)

class RealtimeInferencePipeline:

    def __init__(self, model_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 batch_size: int = 8, frame_buffer_size: int = 30):

        self.device = torch.device(device)
        self.batch_size = batch_size
        self.frame_buffer_size = frame_buffer_size

        self.model = self._load_model(model_path)
        self.model.eval()

        self.inference_times = []
        self.throughput_fps = 0

    def _load_model(self, model_path: str) -> torch.nn.Module:

        try:
            from src.models.transformer_models import TransformerBehaviorPredictor
            model = TransformerBehaviorPredictor(input_dim=6, d_model=256, num_heads=8, num_layers=4)
            
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            else:
                model = checkpoint
            
            model.to(self.device)
            logger.info(f"Loaded model from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def preprocess_trajectories(self, trajectories: Dict) -> Tuple[torch.Tensor, torch.Tensor]:

        if not trajectories:
            return torch.tensor([]), torch.tensor([])

        features = []
        for person_id, traj in trajectories.items():
            points = traj.points
            if len(points) < 5:
                continue

            recent = points[-self.frame_buffer_size:]

            feature_list = []
            for p in recent:
                feature_list.append([p.x, p.y, p.vx, p.vy, p.ax, p.ay])

            if len(feature_list) < self.frame_buffer_size:
                pad_len = self.frame_buffer_size - len(feature_list)
                pad = np.zeros((pad_len, 6))
                feature_list = np.vstack([pad, feature_list])

            features.append(feature_list)

        if not features:
            return torch.tensor([]), torch.tensor([])

        features = np.array(features)
        features = torch.from_numpy(features).float().to(self.device)

        return features

    def process_video(self, video_path: str, skip_frames: int = 1) -> Dict:

        from src.data_pipeline.video_loader import VideoLoader
        from src.data_pipeline.trajectory_extractor import TrajectoryExtractor

        loader = VideoLoader()
        extractor = TrajectoryExtractor()

        results = {
            'anomaly_scores': [],
            'trajectories': [],
            'frames': [],
            'timestamps': []
        }

        start_time = time.time()
        frame_count = 0

        try:
            for frame_idx, frame in loader.extract_frames(video_path, frame_interval=skip_frames):

                detections = extractor.detector.detect(frame)
                trajectories = extractor.tracker.update(detections, frame_idx)

                if trajectories:
                    features = self.preprocess_trajectories(trajectories)
                    if len(features) > 0:
                        with torch.no_grad():
                            start_inf = time.time()
                            scores, _ = self.model(features)
                            inf_time = (time.time() - start_inf) * 1000
                            self.inference_times.append(inf_time)

                        results['anomaly_scores'].append(scores.cpu().numpy())
                        results['trajectories'].append(trajectories)
                        results['frames'].append(frame)
                        results['timestamps'].append(frame_idx)

                frame_count += 1
        finally:
            extractor.tracker.reset()

        total_time = time.time() - start_time
        self.throughput_fps = frame_count / total_time

        logger.info(f"Processed {frame_count} frames in {total_time:.2f}s")
        logger.info(f"Throughput: {self.throughput_fps:.2f} fps")
        if self.inference_times:
            logger.info(f"Avg inference time: {np.mean(self.inference_times):.2f}ms")

        return results

    def predict_single_frame(self, trajectories: Dict) -> Tuple[np.ndarray, np.ndarray]:

        features = self.preprocess_trajectories(trajectories)

        if len(features) == 0:
            return np.array([]), np.array([])

        with torch.no_grad():
            anomaly_scores, attention = self.model(features)

        return anomaly_scores.cpu().numpy(), attention.cpu().numpy()

class BatchInferencePipeline:

    def __init__(self, model_path: str, device: str = "cuda", batch_size: int = 32):
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.model = self._load_model(model_path)
        self.model.eval()

    def _load_model(self, model_path: str) -> torch.nn.Module:
        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model = checkpoint['model_state_dict']
        else:
            model = checkpoint
        return model.to(self.device)

    def predict_batch(self, features: np.ndarray) -> np.ndarray:

        features = torch.from_numpy(features).float().to(self.device)

        with torch.no_grad():
            predictions, _ = self.model(features)

        return predictions.cpu().numpy()

    def benchmark(self, features: np.ndarray, num_iterations: int = 100) -> Dict:

        times = []

        for _ in range(num_iterations):
            features_batch = torch.from_numpy(features).float().to(self.device)

            start = time.time()
            with torch.no_grad():
                _ = self.model(features_batch)
            elapsed = (time.time() - start) * 1000

            times.append(elapsed)

        times = np.array(times[10:])

        return {
            'mean_latency_ms': np.mean(times),
            'median_latency_ms': np.median(times),
            'min_latency_ms': np.min(times),
            'max_latency_ms': np.max(times),
            'std_latency_ms': np.std(times),
            'p95_latency_ms': np.percentile(times, 95),
            'p99_latency_ms': np.percentile(times, 99),
            'throughput_fps': 1000 / np.mean(times) * self.batch_size
        }

class EnsemblePredictor:

    def __init__(self, model_paths: List[str], device: str = "cuda"):

        self.device = torch.device(device)
        self.models = []

        for path in model_paths:
            model = torch.load(path, map_location=self.device)
            model.eval()
            self.models.append(model)

    def predict(self, features: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:

        predictions = []
        attentions = []

        for model in self.models:
            with torch.no_grad():
                pred, attn = model(features)
                predictions.append(pred.cpu().numpy())
                attentions.append(attn.cpu().numpy())

        ensemble_pred = np.mean(predictions, axis=0)
        ensemble_attn = np.mean(attentions, axis=0)

        return ensemble_pred, ensemble_attn

if __name__ == "__main__":

    import sys

    model_path = "models/checkpoints/model.pt"
    video_path = "data/raw/sample.mp4"

    if not Path(model_path).exists():
        logger.warning(f"Model not found at {model_path}")

    if not Path(video_path).exists():
        logger.warning(f"Video not found at {video_path}")
    else:
        pipeline = RealtimeInferencePipeline(model_path)
        results = pipeline.process_video(video_path)
        print(f"Processed video: {len(results['anomaly_scores'])} predictions")
