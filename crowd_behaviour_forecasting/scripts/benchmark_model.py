import numpy as np
import torch
import logging
import time
import json
from pathlib import Path
from typing import Dict, List
import argparse

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class LatencyBenchmark:

    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = torch.device(device)
        self.model = torch.load(model_path, map_location=self.device)
        self.model.eval()

    def benchmark_throughput(self, batch_sizes: List[int] = None,
                           num_iterations: int = 100) -> Dict:

        if batch_sizes is None:
            batch_sizes = [1, 4, 8, 16, 32]

        results = {}

        for batch_size in batch_sizes:
            logger.info(f"Benchmarking batch_size={batch_size}")

            x = torch.randn(batch_size, 30, 6, device=self.device)

            with torch.no_grad():
                for _ in range(10):
                    _ = self.model(x)

            times = []
            with torch.no_grad():
                for _ in range(num_iterations):
                    start = time.time()
                    _ = self.model(x)
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    times.append(time.time() - start)

            times = np.array(times[10:]) * 1000

            results[f"batch_{batch_size}"] = {
                'mean_latency_ms': float(np.mean(times)),
                'median_latency_ms': float(np.median(times)),
                'p95_latency_ms': float(np.percentile(times, 95)),
                'p99_latency_ms': float(np.percentile(times, 99)),
                'throughput_fps': float(batch_size * 1000 / np.mean(times))
            }

        return results

    def benchmark_memory(self, batch_sizes: List[int] = None) -> Dict:

        if batch_sizes is None:
            batch_sizes = [1, 4, 8, 16, 32]

        results = {}

        for batch_size in batch_sizes:
            logger.info(f"Benchmarking memory for batch_size={batch_size}")

            torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

            x = torch.randn(batch_size, 30, 6, device=self.device)

            with torch.no_grad():
                _ = self.model(x)

            if torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
                results[f"batch_{batch_size}"] = {'peak_memory_mb': float(peak_memory)}

        return results

def run_benchmarks(model_path: str, output_dir: str = "results/benchmarks"):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not Path(model_path).exists():
        logger.warning(f"Model not found: {model_path}. Skipping benchmarks.")
        return

    benchmark = LatencyBenchmark(model_path)

    logger.info("Running throughput benchmarks...")
    throughput_results = benchmark.benchmark_throughput()

    logger.info("Running memory benchmarks...")
    memory_results = benchmark.benchmark_memory()

    results = {
        'model': model_path,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'throughput': throughput_results,
        'memory': memory_results
    }

    output_file = output_dir / "benchmark_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {output_file}")

    print("\n" + "="*50)
    print("BENCHMARK SUMMARY")
    print("="*50)
    print("\nThroughput Results:")
    for batch_key, metrics in throughput_results.items():
        print(f"\n{batch_key}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.2f}")

    if memory_results:
        print("\nMemory Results:")
        for batch_key, metrics in memory_results.items():
            print(f"\n{batch_key}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark crowd forecasting model")
    parser.add_argument("--model", default="models/checkpoints/best.pt",
                       help="Model path")
    parser.add_argument("--output", default="results/benchmarks",
                       help="Output directory")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device")

    args = parser.parse_args()

    run_benchmarks(args.model, args.output)
