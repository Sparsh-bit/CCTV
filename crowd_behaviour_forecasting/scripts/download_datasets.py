import os
import logging
from pathlib import Path
import urllib.request
import zipfile
import tarfile

logger = logging.getLogger(__name__)

class DatasetManager:

    DATASETS = {
        'umn': {
            'urls': [
                'http://mha.cs.umn.edu/movies/crowdactivity/videos/raw_video/Raw-Raw.avi',
                'http://mha.cs.umn.edu/movies/crowdactivity/videos/abnormality_gt.rar',
            ],
            'description': 'UMN Crowd Anomaly Dataset',
            'size': '~5GB',
            'format': 'AVI + Ground Truth',
        },
        'shanghaitech': {
            'urls': [
                'https://github.com/muhammadehsan/Anomaly-Detection-and-Localization/releases/download/v1.0/ShanghaiTech_Anomaly_Dataset.zip'
            ],
            'description': 'ShanghaiTech Campus Anomaly Detection',
            'size': '~3GB',
            'format': 'MP4',
        },
        'ucf_crime': {
            'urls': [
                'http://crcv.ucf.edu/projects/real-world/'
            ],
            'description': 'UCF-Crime Dataset',
            'size': '~150GB',
            'format': 'MP4',
            'note': 'Requires manual download due to size'
        }
    }

    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def list_datasets(self):

        print("\nAvailable Datasets:\n")
        for name, info in self.DATASETS.items():
            print(f"{name.upper()}")
            print(f"  Description: {info['description']}")
            print(f"  Size: {info['size']}")
            print(f"  Format: {info['format']}")
            if 'note' in info:
                print(f"  Note: {info['note']}")
            print()

    def download_umn(self):

        print("Downloading UMN Dataset...")
        print("\nIMPORTANT: Please visit http://mha.cs.umn.edu/movies/ and download manually")
        print("Place files in: data/raw/umn/")
        print("\nDataset Information:")
        print("- Multiple locations with abnormal crowd behavior")
        print("- Running, panic behavior")
        print("- Frame-level ground truth")

        return False

    def download_shanghaitech(self):

        dataset_dir = self.data_dir / "shanghaitech"
        dataset_dir.mkdir(exist_ok=True)

        print("Downloading ShanghaiTech Dataset...")

        try:
            url = self.DATASETS['shanghaitech']['urls'][0]
            zip_path = dataset_dir / "shanghaitech.zip"

            print(f"Downloading from {url}...")
            urllib.request.urlretrieve(url, zip_path, self._download_progress)

            print(f"\nExtracting to {dataset_dir}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(dataset_dir)

            zip_path.unlink()
            print(f"Download complete: {dataset_dir}")
            return True

        except Exception as e:
            logger.error(f"Download failed: {e}")
            return False

    def download_ucf_crime(self):

        print("UCF-Crime Dataset Download")
        print("\nIMPORTANT: This dataset requires manual download")
        print("URL: http://crcv.ucf.edu/projects/real-world/")
        print("\nSteps:")
        print("1. Visit the website above")
        print("2. Request dataset access")
        print("3. Download to: data/raw/ucf_crime/")
        print("\nNote: Dataset is ~150GB, so consider using a subset for testing")

        return False

    def download_dataset(self, dataset_name: str) -> bool:

        if dataset_name.lower() not in self.DATASETS:
            print(f"Unknown dataset: {dataset_name}")
            return False

        if dataset_name.lower() == 'umn':
            return self.download_umn()
        elif dataset_name.lower() == 'shanghaitech':
            return self.download_shanghaitech()
        elif dataset_name.lower() == 'ucf_crime':
            return self.download_ucf_crime()

        return False

    @staticmethod
    def _download_progress(block_num, block_size, total_size):

        downloaded = block_num * block_size
        percent = min(downloaded * 100 // total_size, 100)
        print(f"\rDownload progress: {percent}%", end="")

class SyntheticDataGenerator:

    @staticmethod
    def generate_synthetic_video(output_path: str, duration: int = 60,
                                fps: int = 30, width: int = 1280,
                                height: int = 720, num_agents: int = 20):

        import cv2
        import numpy as np

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        agents = np.random.rand(num_agents, 4) * np.array([width, height, 10, 10])

        print(f"Generating synthetic video: {output_path}")

        for frame_idx in range(duration * fps):

            frame = np.ones((height, width, 3), dtype=np.uint8) * 50

            agents[:, 0] += agents[:, 2]
            agents[:, 1] += agents[:, 3]

            agents[:, 0] = np.where(agents[:, 0] > width, width, agents[:, 0])
            agents[:, 0] = np.where(agents[:, 0] < 0, 0, agents[:, 0])
            agents[:, 1] = np.where(agents[:, 1] > height, height, agents[:, 1])
            agents[:, 1] = np.where(agents[:, 1] < 0, 0, agents[:, 1])

            agents[agents[:, 0] > width, 2] *= -1
            agents[agents[:, 0] < 0, 2] *= -1
            agents[agents[:, 1] > height, 3] *= -1
            agents[agents[:, 1] < 0, 3] *= -1

            for i, agent in enumerate(agents):
                x, y = int(agent[0]), int(agent[1])
                color = (0, 255, 0) if i % 5 != 0 else (0, 0, 255)
                cv2.circle(frame, (x, y), 5, color, -1)

            cv2.putText(frame, f"Frame: {frame_idx}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Agents: {num_agents}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            out.write(frame)

            if (frame_idx + 1) % 30 == 0:
                print(f"  Frames: {frame_idx + 1}/{duration * fps}")

        out.release()
        print(f"Video saved: {output_path}")
        return str(output_path)

def setup_datasets():

    base_dir = Path("data/raw")
    base_dir.mkdir(parents=True, exist_ok=True)

    for dataset in ["umn", "shanghaitech", "ucf_crime", "synthetic"]:
        (base_dir / dataset).mkdir(exist_ok=True)

    print(f"Dataset directories created in {base_dir}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download datasets")
    parser.add_argument("--dataset", choices=['umn', 'shanghaitech', 'ucf_crime', 'all', 'list'],
                       help="Dataset to download")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic video")
    parser.add_argument("--duration", type=int, default=60, help="Synthetic video duration")

    args = parser.parse_args()

    setup_datasets()

    manager = DatasetManager()

    if args.dataset == 'list':
        manager.list_datasets()
    elif args.synthetic:
        SyntheticDataGenerator.generate_synthetic_video(
            "data/raw/synthetic/sample.mp4",
            duration=args.duration
        )
    elif args.dataset:
        if args.dataset == 'all':
            for dataset in ['umn', 'shanghaitech', 'ucf_crime']:
                manager.download_dataset(dataset)
        else:
            manager.download_dataset(args.dataset)
    else:
        parser.print_help()
