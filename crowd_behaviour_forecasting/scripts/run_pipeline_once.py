"""Quick runner to execute the RealtimeInferencePipeline on a sample video.
Generates results JSON, heatmap PNG, and overlay MP4 in the `results/` directory.
"""
import sys
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.inference.inference_pipeline import RealtimeInferencePipeline
from src.data_pipeline.video_loader import VideoLoader
from src.api.backend import RESULTS_DIR

def main():
    sample_video = Path('data/raw/synthetic/sample.mp4')
    if not sample_video.exists():
        print('Sample video not found at', sample_video)
        return

    RESULTS_DIR.mkdir(exist_ok=True)

    pipeline = RealtimeInferencePipeline(model_path='models/checkpoints/transformer_final.pt', device='cpu', batch_size=8)
    results = pipeline.process_video(str(sample_video), skip_frames=5)

    out_json = RESULTS_DIR / 'quick_pipeline_result.json'
    with open(out_json, 'w') as f:
        json.dump({'num_frames': len(results.get('frames', [])), 'anomaly_scores_len': len(results.get('anomaly_scores', []))}, f, indent=2)

    print('Pipeline finished. Frames processed:', len(results.get('frames', [])))

if __name__ == '__main__':
    main()
