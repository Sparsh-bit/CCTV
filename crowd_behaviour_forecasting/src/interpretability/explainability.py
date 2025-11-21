import numpy as np
import cv2
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import logging

logger = logging.getLogger(__name__)

class AnomalyHeatmapGenerator:

    def __init__(self, frame_size: Tuple[int, int] = (1280, 720), sigma: float = 20.0):

        self.frame_size = frame_size
        self.sigma = sigma

    def generate_heatmap(self, trajectories: Dict, anomaly_scores: np.ndarray) -> np.ndarray:

        heatmap = np.zeros(self.frame_size, dtype=np.float32)

        if not trajectories:
            return heatmap

        for idx, (person_id, traj) in enumerate(trajectories.items()):
            if idx < len(anomaly_scores):
                score = float(anomaly_scores[idx][0]) if len(anomaly_scores[idx].shape) > 0 else float(anomaly_scores[idx])

                if len(traj.points) > 0:
                    last_point = traj.points[-1]
                    x, y = int(last_point.x), int(last_point.y)

                    x = np.clip(x, 0, self.frame_size[0] - 1)
                    y = np.clip(y, 0, self.frame_size[1] - 1)

                    heatmap[y, x] += score

        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        heatmap = cv2.GaussianBlur(heatmap, (31, 31), self.sigma)

        return heatmap

    def colorize_heatmap(self, heatmap: np.ndarray, colormap: str = 'jet') -> np.ndarray:

        cmap = cm.get_cmap(colormap)
        colored = cmap(heatmap)
        colored = (colored[:, :, :3] * 255).astype(np.uint8)
        return cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)

    def overlay_heatmap(self, frame: np.ndarray, heatmap: np.ndarray, alpha: float = 0.3) -> np.ndarray:

        heatmap_rgb = self.colorize_heatmap(heatmap)
        heatmap_rgb = cv2.resize(heatmap_rgb, (frame.shape[1], frame.shape[0]))

        result = cv2.addWeighted(frame, 1 - alpha, heatmap_rgb, alpha, 0)
        return result

class AttentionVisualizer:

    @staticmethod
    def visualize_trajectory_attention(frame: np.ndarray, trajectories: Dict,
                                      attention_weights: np.ndarray) -> np.ndarray:

        vis = frame.copy()

        for idx, (person_id, traj) in enumerate(trajectories.items()):
            if idx < len(attention_weights):
                weight = float(attention_weights[idx].mean())

                color_intensity = int(255 * weight)
                color = (0, color_intensity, 255 - color_intensity)

                if len(traj.points) > 0:
                    last_point = traj.points[-1]
                    x, y = int(last_point.x), int(last_point.y)

                    radius = int(10 + 20 * weight)
                    cv2.circle(vis, (x, y), radius, color, 2)

        return vis

    @staticmethod
    def plot_attention_timeline(attention_weights: np.ndarray) -> np.ndarray:

        fig, ax = plt.subplots(figsize=(12, 4))

        for i in range(attention_weights.shape[0]):
            ax.plot(attention_weights[i, :], label=f"Agent {i}")

        ax.set_xlabel("Time Step")
        ax.set_ylabel("Attention Weight")
        ax.set_title("Temporal Attention Pattern")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)

        return image

class RiskAssessment:

    def __init__(self, thresholds: Dict[str, float] = None):

        self.thresholds = thresholds or {
            'low': 0.3,
            'high': 0.7
        }

    def assess_risk(self, anomaly_scores: np.ndarray) -> Dict:

        if len(anomaly_scores) == 0:
            return {'level': 'none', 'score': 0.0, 'agents': {}}

        mean_score = float(np.mean(anomaly_scores))
        max_score = float(np.max(anomaly_scores))

        if max_score > self.thresholds['high']:
            level = 'high'
        elif max_score > self.thresholds['low']:
            level = 'medium'
        else:
            level = 'low'

        return {
            'level': level,
            'mean_score': mean_score,
            'max_score': max_score,
            'num_anomalies': int(np.sum(anomaly_scores > 0.5))
        }

    def visualize_risk(self, risk_assessment: Dict) -> str:

        level = risk_assessment['level']
        score = risk_assessment['max_score']

        colors = {'low': '🟢', 'medium': '🟡', 'high': '🔴', 'none': '⚪'}

        risk_str = f"{colors.get(level, '❓')} Risk Level: {level.upper()}\n"
        risk_str += f"Max Anomaly Score: {score:.3f}\n"
        risk_str += f"Mean Anomaly Score: {risk_assessment['mean_score']:.3f}\n"
        risk_str += f"Anomalous Agents: {risk_assessment['num_anomalies']}"

        return risk_str

class FeatureImportance:

    @staticmethod
    def compute_gradient_importance(model, features: np.ndarray) -> np.ndarray:

        import torch

        features_tensor = torch.tensor(features, dtype=torch.float32, requires_grad=True)

        outputs, _ = model(features_tensor)
        loss = outputs.mean()

        loss.backward()

        importance = torch.abs(features_tensor.grad).mean(dim=0).detach().numpy()

        return importance

    @staticmethod
    def plot_feature_importance(importance: np.ndarray, feature_names: list = None) -> np.ndarray:

        if feature_names is None:
            feature_names = ['x', 'y', 'vx', 'vy', 'ax', 'ay']

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(feature_names, importance)

        colors = plt.cm.viridis(importance / importance.max())
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        ax.set_xlabel("Importance")
        ax.set_title("Feature Importance for Anomaly Detection")
        ax.grid(True, alpha=0.3, axis='x')

        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)

        return image

class AnomalyVisualizer:

    def __init__(self, frame_size: Tuple[int, int] = (1280, 720)):
        self.frame_size = frame_size
        self.heatmap_gen = AnomalyHeatmapGenerator(frame_size)
        self.attention_viz = AttentionVisualizer()
        self.risk_assess = RiskAssessment()

    def visualize_complete(self, frame: np.ndarray, trajectories: Dict,
                          anomaly_scores: np.ndarray, attention_weights: np.ndarray,
                          frame_idx: int = 0) -> np.ndarray:

        vis = frame.copy()

        heatmap = self.heatmap_gen.generate_heatmap(trajectories, anomaly_scores)
        vis = self.heatmap_gen.overlay_heatmap(vis, heatmap, alpha=0.4)

        for idx, (person_id, traj) in enumerate(trajectories.items()):
            if len(traj.points) > 0:
                weight = float(attention_weights[idx].mean()) if idx < len(attention_weights) else 0.5
                score = float(anomaly_scores[idx][0]) if len(anomaly_scores[idx].shape) > 0 else 0.0

                color_intensity = int(255 * weight)
                color = (0, color_intensity, 255 - color_intensity)

                last_point = traj.points[-1]
                x, y = int(last_point.x), int(last_point.y)

                cv2.circle(vis, (x, y), 8, color, -1)

                text = f"P{person_id}:{score:.2f}"
                cv2.putText(vis, text, (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        risk = self.risk_assess.assess_risk(anomaly_scores)
        risk_text = f"Risk: {risk['level'].upper()} | Score: {risk['max_score']:.3f}"
        cv2.putText(vis, risk_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(vis, f"Frame: {frame_idx}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)

        return vis

    def save_visualization(self, output_path: str, frame: np.ndarray):

        cv2.imwrite(output_path, frame)
        logger.info(f"Saved visualization to {output_path}")

if __name__ == "__main__":

    frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    heatmap_gen = AnomalyHeatmapGenerator()

    heatmap = np.random.rand(720, 1280)
    colored = heatmap_gen.colorize_heatmap(heatmap)

    print(f"Heatmap shape: {heatmap.shape}")
    print(f"Colored heatmap shape: {colored.shape}")
