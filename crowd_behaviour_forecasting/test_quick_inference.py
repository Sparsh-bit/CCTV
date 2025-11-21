import torch
import numpy as np
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def quick_inference_test():
    
    from src.models.transformer_models import TransformerBehaviorPredictor
    
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
    else:
        model = checkpoint
    
    model.to(device)
    model.eval()
    
    logger.info("Model loaded successfully!")
    logger.info(f"Model device: {next(model.parameters()).device}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    logger.info("\nTesting inference...")
    with torch.no_grad():
        test_data = torch.randn(4, 30, 6).to(device)
        
        anomaly_scores, attention = model(test_data)
        logger.info(f"Input shape: {test_data.shape}")
        logger.info(f"Anomaly scores shape: {anomaly_scores.shape}")
        logger.info(f"Attention shape: {attention.shape}")
        logger.info(f"Anomaly scores: {anomaly_scores}")
    
    results = {
        'status': 'success',
        'model_path': model_path,
        'device': str(device),
        'test_input_shape': [4, 30, 6],
        'anomaly_scores_shape': list(anomaly_scores.shape),
        'attention_shape': list(attention.shape),
        'anomaly_scores_sample': anomaly_scores.cpu().numpy().tolist()
    }
    
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / "quick_inference_test.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Test results saved to {output_path}")
    logger.info("QUICK INFERENCE TEST COMPLETED SUCCESSFULLY!")
    
    return results

if __name__ == "__main__":
    try:
        quick_inference_test()
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
