"""Test script to instantiate the RealtimeInferencePipeline and print brief status."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.inference.inference_pipeline import RealtimeInferencePipeline

def main():
    print('Instantiating pipeline...')
    pipeline = RealtimeInferencePipeline(model_path='models/checkpoints/transformer_final.pt', device='cpu', batch_size=4)
    print('Model device:', pipeline.device)
    print('Model eval mode:', pipeline.model.training)
    print('Done')

if __name__ == '__main__':
    main()
