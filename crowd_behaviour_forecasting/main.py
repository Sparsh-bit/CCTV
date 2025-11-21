import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_command(args):

    from scripts.download_datasets import setup_datasets, SyntheticDataGenerator

    setup_datasets()

    if args.synthetic:
        logger.info("Generating synthetic video for testing...")
        SyntheticDataGenerator.generate_synthetic_video(
            "data/raw/synthetic/sample.mp4",
            duration=args.duration
        )

    logger.info("Setup complete")

def extract_command(args):

    from src.data_pipeline.trajectory_extractor import TrajectoryExtractor
    import json

    extractor = TrajectoryExtractor(detector_type="yolov8", model_size=args.model_size)
    trajectories = extractor.extract_from_video(args.video)

    output_path = Path(args.output) / f"{Path(args.video).stem}_trajectories.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    traj_data = {}
    for person_id, traj in trajectories.items():
        traj_data[person_id] = {
            'points': [
                {'frame': p.frame_id, 'x': p.x, 'y': p.y, 'vx': p.vx, 'vy': p.vy}
                for p in traj.points
            ]
        }

    with open(output_path, 'w') as f:
        json.dump(traj_data, f, indent=2)

    logger.info(f"Trajectories saved to {output_path}")

def train_command(args):

    from src.models.train import TrainingManager
    import torch

    manager = TrainingManager(args.config, args.model_type)
    model = manager.build_model()

    dataset = manager.create_dummy_dataset(num_samples=100)

    manager.train(model, dataset, num_epochs=args.epochs, batch_size=args.batch_size)

    output_path = Path("models/checkpoints") / f"{args.model_type}_final.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    logger.info(f"Model saved to {output_path}")

def infer_command(args):

    from src.inference.inference_pipeline import RealtimeInferencePipeline
    from src.interpretability.explainability import AnomalyVisualizer
    import cv2

    pipeline = RealtimeInferencePipeline(args.model, device=args.device)

    results = pipeline.process_video(args.video)

    logger.info(f"Processed {len(results['anomaly_scores'])} frames")
    logger.info(f"Average inference time: {np.mean(pipeline.inference_times):.2f}ms")

def deploy_command(args):

    from src.edge_deployment.optimization import EdgeModelDeployment

    deployment = EdgeModelDeployment(args.model, args.output_dir)

    info = deployment.deploy(
        quantize=args.quantize,
        tensorrt=args.tensorrt
    )

    logger.info(f"Deployment files: {info}")

def server_command(args):

    import subprocess
    import sys

    cmd = [
        sys.executable, "-m", "uvicorn",
        "src.edge_deployment.api_server:app",
        "--host", args.host,
        "--port", str(args.port)
    ]

    if args.reload:
        cmd.append("--reload")

    if args.workers:
        cmd.extend(["--workers", str(args.workers)])

    logger.info(f"Starting server: {' '.join(cmd)}")
    subprocess.run(cmd)

def benchmark_command(args):

    from scripts.benchmark_model import run_benchmarks

    run_benchmarks(args.model, args.output)

def main():

    parser = argparse.ArgumentParser(
        description="Crowd Behaviour Forecasting for Smart Cities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  python main.py setup --synthetic --duration 120

  python main.py extract --video data/raw/video.mp4 --output data/processed/

  python main.py train --config configs/model_config.yaml --model_type gnn --epochs 100

  python main.py infer --video data/raw/test.mp4 --model models/checkpoints/best.pt

  python main.py deploy --model models/checkpoints/best.pt --quantize --tensorrt

  python main.py server --port 8000 --workers 4

  python main.py benchmark --model models/checkpoints/best.pt
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command')

    setup_parser = subparsers.add_parser('setup', help='Setup environment')
    setup_parser.add_argument('--synthetic', action='store_true',
                             help='Generate synthetic video')
    setup_parser.add_argument('--duration', type=int, default=60,
                             help='Synthetic video duration in seconds')
    setup_parser.set_defaults(func=setup_command)

    extract_parser = subparsers.add_parser('extract', help='Extract trajectories')
    extract_parser.add_argument('--video', required=True, help='Video file path')
    extract_parser.add_argument('--output', default='data/processed',
                               help='Output directory')
    extract_parser.add_argument('--model_size', default='m',
                               choices=['n', 's', 'm', 'l', 'x'],
                               help='YOLOv8 model size')
    extract_parser.set_defaults(func=extract_command)

    train_parser = subparsers.add_parser('train', help='Train model')
    train_parser.add_argument('--config', default='configs/model_config.yaml',
                             help='Config file path')
    train_parser.add_argument('--model_type', default='gnn',
                             choices=['gnn', 'transformer', 'convlstm'],
                             help='Model type')
    train_parser.add_argument('--epochs', type=int, default=100,
                             help='Number of epochs')
    train_parser.add_argument('--batch_size', type=int, default=32,
                             help='Batch size')
    train_parser.set_defaults(func=train_command)

    infer_parser = subparsers.add_parser('infer', help='Run inference')
    infer_parser.add_argument('--video', required=True, help='Video file path')
    infer_parser.add_argument('--model', required=True, help='Model path')
    infer_parser.add_argument('--device', default='cuda',
                             choices=['cuda', 'cpu'],
                             help='Device')
    infer_parser.set_defaults(func=infer_command)

    deploy_parser = subparsers.add_parser('deploy', help='Deploy model')
    deploy_parser.add_argument('--model', required=True, help='Model path')
    deploy_parser.add_argument('--output_dir', default='models/deployment',
                              help='Output directory')
    deploy_parser.add_argument('--quantize', action='store_true',
                              help='Quantize model')
    deploy_parser.add_argument('--tensorrt', action='store_true',
                              help='Use TensorRT optimization')
    deploy_parser.set_defaults(func=deploy_command)

    server_parser = subparsers.add_parser('server', help='Start API server')
    server_parser.add_argument('--host', default='0.0.0.0',
                              help='Server host')
    server_parser.add_argument('--port', type=int, default=8000,
                              help='Server port')
    server_parser.add_argument('--workers', type=int,
                              help='Number of workers')
    server_parser.add_argument('--reload', action='store_true',
                              help='Auto-reload on changes')
    server_parser.set_defaults(func=server_command)

    benchmark_parser = subparsers.add_parser('benchmark', help='Benchmark model')
    benchmark_parser.add_argument('--model', required=True, help='Model path')
    benchmark_parser.add_argument('--output', default='results/benchmarks',
                                 help='Output directory')
    benchmark_parser.set_defaults(func=benchmark_command)

    args = parser.parse_args()

    if hasattr(args, 'func'):
        try:
            args.func(args)
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            sys.exit(1)
    else:
        parser.print_help()

if __name__ == '__main__':
    import numpy as np
    main()
