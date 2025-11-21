"""
Unit tests for crowd behavior forecasting
"""
import pytest
import torch
import numpy as np
from pathlib import Path


class TestDataPipeline:
    """Test data pipeline components."""
    
    def test_video_loader(self):
        """Test video loading."""
        # Skip if no test video available
        pytest.skip("No test video available")
    
    def test_trajectory_extraction(self):
        """Test trajectory extraction."""
        pytest.skip("YOLOv8 not loaded in test environment")
    
    def test_frame_buffer(self):
        """Test frame buffer."""
        from src.data_pipeline.video_loader import FrameBuffer
        
        buffer = FrameBuffer(buffer_size=10, frame_shape=(720, 1280))
        
        # Add frames
        for i in range(5):
            frame = np.random.rand(720, 1280, 3)
            result = buffer.add_frame(frame)
            assert result is None  # Buffer not full
        
        # Fill buffer
        for i in range(5, 10):
            frame = np.random.rand(720, 1280, 3)
            result = buffer.add_frame(frame)
            if i == 9:
                assert result is not None
                assert result.shape == (10, 720, 1280, 3)


class TestModels:
    """Test model architectures."""
    
    def test_gnn_model(self):
        """Test GNN model."""
        from src.models.gnn_models import SpatioTemporalGCN
        
        model = SpatioTemporalGCN(input_dim=6, hidden_dim=64, num_layers=2)
        
        # Test forward pass
        x = torch.randn(10, 6)  # 10 nodes, 6 features
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])  # 3 edges
        
        scores, attn = model(x, edge_index)
        
        assert scores.shape == (10, 1)
        assert attn.shape == (10, 64)
    
    def test_transformer_model(self):
        """Test Transformer model."""
        from src.models.transformer_models import TransformerBehaviorPredictor
        
        model = TransformerBehaviorPredictor(
            input_dim=6,
            d_model=128,
            num_heads=4,
            num_layers=2
        )
        
        # Test forward pass
        x = torch.randn(4, 30, 6)  # batch=4, seq_len=30, features=6
        scores, attn = model(x)
        
        assert scores.shape == (4, 1)
        assert attn.shape == (4, 30, 1)
    
    def test_convlstm_model(self):
        """Test ConvLSTM model."""
        from src.models.transformer_models import ConvLSTMBehaviorDetector
        
        model = ConvLSTMBehaviorDetector(
            input_channels=3,
            hidden_channels=32,
            num_layers=2
        )
        
        # Test forward pass
        x = torch.randn(2, 10, 3, 64, 64)  # batch=2, time=10, channels=3, h=64, w=64
        scores, attn = model(x)
        
        assert scores.shape[0] == 2  # batch size


class TestInference:
    """Test inference pipeline."""
    
    def test_batch_inference(self):
        """Test batch inference."""
        pytest.skip("Model not available in test environment")
    
    def test_ensemble_predictor(self):
        """Test ensemble predictions."""
        pytest.skip("Models not available in test environment")


class TestInterpretability:
    """Test interpretability modules."""
    
    def test_heatmap_generation(self):
        """Test heatmap generation."""
        from src.interpretability.explainability import AnomalyHeatmapGenerator
        
        gen = AnomalyHeatmapGenerator(frame_size=(640, 480))
        
        # Test with empty trajectories
        heatmap = gen.generate_heatmap({}, np.array([]))
        assert heatmap.shape == (480, 640)
        assert heatmap.max() <= 1.0
    
    def test_risk_assessment(self):
        """Test risk assessment."""
        from src.interpretability.explainability import RiskAssessment
        
        assessor = RiskAssessment()
        
        # Low risk
        scores = np.array([[0.2], [0.1]])
        risk = assessor.assess_risk(scores)
        assert risk['level'] == 'low'
        
        # High risk
        scores = np.array([[0.8], [0.9]])
        risk = assessor.assess_risk(scores)
        assert risk['level'] == 'high'


class TestDeployment:
    """Test deployment utilities."""
    
    def test_model_info(self):
        """Test model info extraction."""
        pytest.skip("Model files not available in test environment")
    
    def test_memory_estimation(self):
        """Test memory usage estimation."""
        pytest.skip("Model files not available in test environment")


class TestUtils:
    """Test utility functions."""
    
    def test_config_loading(self):
        """Test configuration loading."""
        import yaml
        from pathlib import Path
        
        config_path = Path("configs/model_config.yaml")
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)
            
            assert 'model' in config
            assert 'training' in config
            assert 'deployment' in config
    
    def test_dataset_manager(self):
        """Test dataset manager."""
        from scripts.download_datasets import DatasetManager
        
        manager = DatasetManager()
        datasets = list(manager.DATASETS.keys())
        
        assert 'umn' in datasets
        assert 'shanghaitech' in datasets
        assert 'ucf_crime' in datasets


# Fixtures
@pytest.fixture
def dummy_model():
    """Fixture providing dummy model."""
    from src.models.gnn_models import SpatioTemporalGCN
    return SpatioTemporalGCN(input_dim=6, hidden_dim=64)


@pytest.fixture
def dummy_trajectories():
    """Fixture providing dummy trajectories."""
    from src.data_pipeline.trajectory_extractor import Trajectory, TrajectoryPoint
    
    trajectories = {}
    for person_id in range(5):
        traj = Trajectory(person_id=person_id)
        for frame_id in range(30):
            point = TrajectoryPoint(
                frame_id=frame_id,
                x=100 + frame_id * 5,
                y=100 + frame_id * 3
            )
            traj.add_point(point)
        trajectories[person_id] = traj
    
    return trajectories


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
