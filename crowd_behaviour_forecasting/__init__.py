"""
Crowd Behaviour Forecasting Package
"""
__version__ = "1.0.0"
__author__ = "Smart Cities AI Lab"

# Make submodules accessible
from src.data_pipeline.video_loader import VideoLoader
from src.data_pipeline.trajectory_extractor import TrajectoryExtractor

__all__ = [
    "VideoLoader",
    "TrajectoryExtractor",
]
