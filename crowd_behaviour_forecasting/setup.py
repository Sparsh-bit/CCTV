from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="crowd-behaviour-forecasting",
    version="1.0.0",
    author="Smart Cities AI Lab",
    author_email="contact@example.com",
    description="Real-time crowd behaviour forecasting with edge deployment for smart cities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/crowd-behaviour-forecasting",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Multimedia :: Video",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "torch-geometric>=2.3.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.11.0",
        "ultralytics>=8.0.0",
        "transformers>=4.33.0",
        "onnx>=1.14.0",
        "onnxruntime>=1.16.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "pydantic>=2.4.0",
        "pyyaml>=6.0.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
        "plotly>=5.17.0",
        "tensorboard>=2.14.0",
        "loguru>=0.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.10.0",
            "flake8>=6.1.0",
            "mypy>=1.6.0",
        ],
        "gpu": [
            "tensorrt>=8.6.0",
            "onnxruntime-gpu>=1.16.0",
        ],
        "inference": [
            "onnxruntime>=1.16.0",
            "onnxruntime-gpu>=1.16.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "crowd-forecast=main:main",
        ],
    },
)
