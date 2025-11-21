import logging
import yaml
from pathlib import Path
from typing import Dict, Any
import numpy as np

logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_config(config: Dict, output_path: str):

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    logger.info(f"Config saved to {output_path}")

def create_logger(name: str, log_file: str = None) -> logging.Logger:

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger

def compute_metrics(predictions: np.ndarray, labels: np.ndarray) -> Dict:

    from sklearn.metrics import (
        precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix, accuracy_score
    )

    preds_binary = (predictions > 0.5).astype(int)

    metrics = {
        'accuracy': accuracy_score(labels, preds_binary),
        'precision': precision_score(labels, preds_binary, zero_division=0),
        'recall': recall_score(labels, preds_binary, zero_division=0),
        'f1': f1_score(labels, preds_binary, zero_division=0),
    }

    try:
        metrics['auc_roc'] = roc_auc_score(labels, predictions)
    except:
        metrics['auc_roc'] = 0.0

    return metrics

def normalize_trajectories(trajectories: np.ndarray,
                          mean: np.ndarray = None,
                          std: np.ndarray = None) -> np.ndarray:

    if mean is None:
        mean = trajectories.mean(axis=(0, 1), keepdims=True)
    if std is None:
        std = trajectories.std(axis=(0, 1), keepdims=True)

    return (trajectories - mean) / (std + 1e-8)

def denormalize_trajectories(trajectories: np.ndarray,
                            mean: np.ndarray,
                            std: np.ndarray) -> np.ndarray:

    return trajectories * std + mean

def get_device():

    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def count_parameters(model) -> int:

    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def format_time(seconds: float) -> str:

    if seconds < 1:
        return f"{seconds*1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        return f"{seconds/60:.1f}m"

class MetricsTracker:

    def __init__(self):
        self.metrics = {}

    def add(self, name: str, value: float):

        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)

    def get_mean(self, name: str) -> float:

        if name not in self.metrics or len(self.metrics[name]) == 0:
            return 0.0
        return np.mean(self.metrics[name])

    def reset(self):

        self.metrics = {}

    def summary(self) -> Dict:

        return {name: self.get_mean(name) for name in self.metrics}

class EarlyStoppingCallback:

    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_value = float('inf')
        self.wait_count = 0
        self.stopped = False

    def check(self, current_value: float) -> bool:

        if current_value < self.best_value - self.min_delta:
            self.best_value = current_value
            self.wait_count = 0
        else:
            self.wait_count += 1

        if self.wait_count >= self.patience:
            self.stopped = True
            return True

        return False

if __name__ == "__main__":

    print("Testing utilities...")

    config = load_config("configs/model_config.yaml")
    print(f"Config loaded: {list(config.keys())}")

    logger = create_logger(__name__, log_file="test.log")
    logger.info("Logger test message")

    print("All tests passed!")
