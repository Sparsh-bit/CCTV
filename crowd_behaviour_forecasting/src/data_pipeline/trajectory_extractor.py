import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
except ImportError:
    logger.warning("ultralytics not installed. Install with: pip install ultralytics")

@dataclass
class Detection:

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int = 0

    @property
    def center(self) -> Tuple[float, float]:

        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def area(self) -> float:

        return (self.x2 - self.x1) * (self.y2 - self.y1)

@dataclass
class TrajectoryPoint:

    frame_id: int
    x: float
    y: float
    x1: float = 0.0
    y1: float = 0.0
    x2: float = 0.0
    y2: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    ax: float = 0.0
    ay: float = 0.0
    confidence: float = 1.0

@dataclass
class Trajectory:

    person_id: int
    points: List[TrajectoryPoint] = field(default_factory=list)

    def add_point(self, point: TrajectoryPoint):

        self.points.append(point)

    def get_velocities(self) -> np.ndarray:

        return np.array([(p.vx, p.vy) for p in self.points])

    def get_accelerations(self) -> np.ndarray:

        return np.array([(p.ax, p.ay) for p in self.points])

    def get_positions(self) -> np.ndarray:

        return np.array([(p.x, p.y) for p in self.points])

    def duration(self) -> int:

        if len(self.points) < 2:
            return 0
        return self.points[-1].frame_id - self.points[0].frame_id

class SimpleTracker:

    def __init__(self, max_disappeared: int = 30, distance_threshold: float = 50):
        self.max_disappeared = max_disappeared
        self.distance_threshold = distance_threshold
        self.next_id = 0
        self.trajectories: Dict[int, Trajectory] = {}
        self.disappeared = {}

    def update(self, detections: List[Detection], frame_id: int) -> Dict[int, Trajectory]:

        if len(detections) == 0:

            for person_id in list(self.disappeared.keys()):
                self.disappeared[person_id] += 1
                if self.disappeared[person_id] > self.max_disappeared:
                    del self.trajectories[person_id]
                    del self.disappeared[person_id]
            return self.trajectories

        centers = np.array([d.center for d in detections])

        used = set()

        for person_id, traj in list(self.trajectories.items()):
            last_point = traj.points[-1]
            last_center = np.array([last_point.x, last_point.y])

            distances = np.linalg.norm(centers - last_center, axis=1)
            closest_idx = np.argmin(distances)
            closest_dist = distances[closest_idx]

            if closest_dist < self.distance_threshold and closest_idx not in used:

                det = detections[closest_idx]
                cx, cy = det.center

                prev_x, prev_y = last_point.x, last_point.y
                vx = cx - prev_x
                vy = cy - prev_y

                ax = vx - last_point.vx if last_point.vx != 0 else 0
                ay = vy - last_point.vy if last_point.vy != 0 else 0

                point = TrajectoryPoint(
                    frame_id=frame_id,
                    x=cx, y=cy,
                    x1=det.x1, y1=det.y1, x2=det.x2, y2=det.y2,
                    vx=vx, vy=vy,
                    ax=ax, ay=ay,
                    confidence=det.confidence
                )
                traj.add_point(point)
                self.disappeared[person_id] = 0
                used.add(closest_idx)

        for idx, det in enumerate(detections):
            if idx not in used:
                cx, cy = det.center
                point = TrajectoryPoint(
                    frame_id=frame_id,
                        x=cx, y=cy,
                        x1=det.x1, y1=det.y1, x2=det.x2, y2=det.y2,
                    confidence=det.confidence
                )
                traj = Trajectory(person_id=self.next_id)
                traj.add_point(point)
                self.trajectories[self.next_id] = traj
                self.disappeared[self.next_id] = 0
                self.next_id += 1

        return self.trajectories

    def reset(self):

        self.trajectories = {}
        self.disappeared = {}
        self.next_id = 0

class YOLOv8Detector:

    def __init__(self, model_size: str = "m", confidence_threshold: float = 0.5):

        self.confidence_threshold = confidence_threshold
        self.model = YOLO(f"yolov8{model_size}.pt")

    def detect(self, frame: np.ndarray) -> List[Detection]:

        results = self.model(frame, verbose=False, conf=self.confidence_threshold)

        detections = []
        for result in results:
            for box in result.boxes:

                if int(box.cls) == 0:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0])

                    det = Detection(
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        confidence=confidence, class_id=0
                    )
                    detections.append(det)

        return detections

class TrajectoryExtractor:

    def __init__(self, detector_type: str = "yolov8", model_size: str = "m"):

        if detector_type == "yolov8":
            self.detector = YOLOv8Detector(model_size=model_size)
        else:
            raise ValueError(f"Unknown detector: {detector_type}")

        self.tracker = SimpleTracker()

    def extract_from_video(self, video_path: str, frame_skip: int = 1) -> Dict[int, Trajectory]:

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        frame_id = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_id % frame_skip == 0:

                    detections = self.detector.detect(frame)

                    self.tracker.update(detections, frame_id)

                frame_id += 1
        finally:
            cap.release()

        trajectories = {
            pid: traj for pid, traj in self.tracker.trajectories.items()
            if len(traj.points) > 5
        }

        logger.info(f"Extracted {len(trajectories)} trajectories from {video_path}")
        return trajectories

    def reset(self):

        self.tracker.reset()

def visualize_trajectories(frame: np.ndarray, trajectories: Dict[int, Trajectory],
                          frame_id: int, history_length: int = 10) -> np.ndarray:

    vis = frame.copy()

    colors = {}
    for person_id, traj in trajectories.items():
        if person_id not in colors:
            colors[person_id] = np.random.randint(0, 255, 3).tolist()

        recent_points = [p for p in traj.points if frame_id - p.frame_id <= history_length]

        if len(recent_points) > 0:

            for i in range(len(recent_points) - 1):
                p1 = recent_points[i]
                p2 = recent_points[i + 1]
                cv2.line(vis, (int(p1.x), int(p1.y)), (int(p2.x), int(p2.y)),
                        colors[person_id], 2)

            last = recent_points[-1]
            cv2.circle(vis, (int(last.x), int(last.y)), 5, colors[person_id], -1)
            cv2.putText(vis, str(person_id), (int(last.x) + 10, int(last.y)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[person_id], 2)

    return vis

if __name__ == "__main__":

    extractor = TrajectoryExtractor(detector_type="yolov8", model_size="m")
    trajectories = extractor.extract_from_video("sample_video.mp4")

    print(f"Extracted {len(trajectories)} trajectories")
    for person_id, traj in trajectories.items():
        print(f"Person {person_id}: {len(traj.points)} frames, "
              f"duration: {traj.duration()} frames")
