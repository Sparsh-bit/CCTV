import cv2
import numpy as np
from typing import Generator, Tuple, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class VideoLoader:

    def __init__(self, target_fps: int = 30, frame_size: Tuple[int, int] = (1280, 720)):
        self.target_fps = target_fps
        self.frame_size = frame_size

    def load_video(self, video_path: str) -> Tuple[int, int, float, int]:

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        cap.release()
        return width, height, fps, frame_count

    def extract_frames(self, video_path: str, frame_interval: Optional[int] = None) -> Generator:

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if frame_interval is None:
            frame_interval = max(1, int(fps / self.target_fps))

        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_interval == 0:

                    frame = cv2.resize(frame, self.frame_size, interpolation=cv2.INTER_LINEAR)
                    yield frame_idx, frame

                frame_idx += 1
        finally:
            cap.release()

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame = frame.astype(np.float32) / 255.0

        return frame

class OpticalFlowProcessor:

    def __init__(self, method: str = "farneback"):
        self.method = method
        self.prev_gray = None

    def compute_flow(self, frame: np.ndarray) -> Optional[np.ndarray]:

        gray = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            return None

        if self.method == "farneback":
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, n8=True, poly_n=5, poly_sigma=1.1, flags=0
            )
        else:
            raise ValueError(f"Unknown optical flow method: {self.method}")

        self.prev_gray = gray
        return flow

    def visualize_flow(self, flow: np.ndarray, frame_shape: Tuple[int, int]) -> np.ndarray:

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        hsv[..., 2] = 255

        flow_viz = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return flow_viz

class FrameNormalizer:

    def __init__(self, mean: Tuple = None, std: Tuple = None):

        self.mean = np.array(mean or [0.485, 0.456, 0.406])
        self.std = np.array(std or [0.229, 0.224, 0.225])

    def normalize(self, frame: np.ndarray) -> np.ndarray:

        frame = (frame - self.mean) / self.std
        return frame

    def denormalize(self, frame: np.ndarray) -> np.ndarray:

        frame = frame * self.std + self.mean
        frame = np.clip(frame, 0, 1)
        return frame

class FrameBuffer:

    def __init__(self, buffer_size: int = 30, frame_shape: Tuple = None):
        self.buffer_size = buffer_size
        self.buffer = []
        self.frame_shape = frame_shape

    def add_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:

        self.buffer.append(frame)

        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

        if len(self.buffer) == self.buffer_size:
            return np.array(self.buffer)
        return None

    def reset(self):

        self.buffer = []

    @property
    def is_full(self) -> bool:

        return len(self.buffer) == self.buffer_size

if __name__ == "__main__":

    loader = VideoLoader(target_fps=30, frame_size=(1280, 720))

    video_path = "sample_video.mp4"
    try:
        for frame_idx, frame in loader.extract_frames(video_path):
            processed = loader.preprocess_frame(frame)
            print(f"Frame {frame_idx}: shape={processed.shape}, dtype={processed.dtype}")

            if frame_idx > 100:
                break
    except Exception as e:
        logger.error(f"Error: {e}")
