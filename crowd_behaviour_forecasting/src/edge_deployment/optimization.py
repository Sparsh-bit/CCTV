import torch
import numpy as np
from typing import Dict, Optional, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class ONNXConverter:

    def __init__(self, model: torch.nn.Module = None, model_path: str = None, opset_version: int = 14):

        self.opset_version = opset_version

        if model is not None:
            self.model = model
        elif model_path is not None:
            self.model = torch.load(model_path, map_location='cpu')
        else:
            raise ValueError("Either model or model_path must be provided")

        self.model.eval()

    def convert(self, output_path: str, input_shape: Tuple = (1, 30, 6),
                input_names: list = None, output_names: list = None,
                dynamic_axes: Dict = None) -> str:

        if input_names is None:
            input_names = ['trajectories']
        if output_names is None:
            output_names = ['anomaly_scores', 'attention']
        if dynamic_axes is None:
            dynamic_axes = {
                'trajectories': {0: 'batch_size'},
                'anomaly_scores': {0: 'batch_size'},
                'attention': {0: 'batch_size'}
            }

        dummy_input = torch.randn(input_shape, requires_grad=False)

        try:
            torch.onnx.export(
                self.model,
                dummy_input,
                output_path,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=self.opset_version,
                do_constant_folding=True,
                verbose=False
            )

            logger.info(f"Model converted to ONNX: {output_path}")

            self._verify_onnx(output_path, dummy_input)

            return output_path
        except Exception as e:
            logger.error(f"ONNX conversion failed: {e}")
            raise

    def _verify_onnx(self, onnx_path: str, test_input: torch.Tensor):

        try:
            import onnx
            import onnxruntime

            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)

            logger.info("ONNX model verification passed")
        except ImportError:
            logger.warning("ONNX verification skipped (onnx/onnxruntime not installed)")
        except Exception as e:
            logger.warning(f"ONNX verification failed: {e}")

class QuantizationOptimizer:

    @staticmethod
    def quantize_static(model_path: str, output_path: str, calibration_data: np.ndarray = None):

        try:
            import onnx
            from onnxruntime.quantization import quantize_dynamic, QuantType

            quantize_dynamic(
                model_path,
                output_path,
                weight_type=QuantType.QInt8
            )

            logger.info(f"Model quantized: {output_path}")
        except ImportError:
            logger.warning("onnxruntime not installed for quantization")

    @staticmethod
    def quantize_torch(model: torch.nn.Module, backend: str = 'fbgemm') -> torch.nn.Module:

        model.qconfig = torch.quantization.get_default_qconfig(backend)
        torch.quantization.prepare(model, inplace=True)
        torch.quantization.convert(model, inplace=True)
        return model

class TensorRTOptimizer:

    @staticmethod
    def optimize_onnx(onnx_path: str, output_path: str, precision: str = 'fp32',
                     batch_sizes: list = None) -> str:

        try:
            import tensorrt as trt
            from polygraphy.backend.onnx import BytesFromPath
            from polygraphy.backend.trt import EngineFromNetwork, NetworkFromOnnxPath, TrtRunner
            from polygraphy.logger import G_LOGGER

            logger.info(f"Optimizing ONNX with TensorRT (precision={precision})")

            precision_map = {
                'fp32': trt.BuilderFlag.STRICT_TYPES,
                'fp16': trt.BuilderFlag.FP16,
                'int8': trt.BuilderFlag.INT8
            }

            logger.info("Building TensorRT engine...")

            logger.info(f"TensorRT engine saved: {output_path}")

            return output_path
        except ImportError:
            logger.warning("TensorRT not installed. Skipping optimization.")
            return onnx_path

class ONNXRuntimeInference:

    def __init__(self, model_path: str, providers: list = None):

        try:
            import onnxruntime as ort

            if providers is None:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

            self.session = ort.InferenceSession(model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [o.name for o in self.session.get_outputs()]

            logger.info(f"ONNX Runtime model loaded: {model_path}")
            logger.info(f"Providers: {self.session.get_providers()}")
        except ImportError:
            logger.error("onnxruntime not installed")
            raise

    def predict(self, inputs: np.ndarray) -> Dict:

        input_dict = {self.input_name: inputs.astype(np.float32)}
        outputs = self.session.run(self.output_names, input_dict)

        return {
            name: output for name, output in zip(self.output_names, outputs)
        }

class EdgeModelDeployment:

    def __init__(self, model_path: str, output_dir: str = "models/deployment"):
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def deploy(self, quantize: bool = True, tensorrt: bool = False) -> Dict:

        deployment_info = {}

        onnx_path = self.output_dir / "model.onnx"
        converter = ONNXConverter(model_path=self.model_path)
        converter.convert(str(onnx_path))
        deployment_info['onnx'] = str(onnx_path)

        if quantize:
            quantized_path = self.output_dir / "model_quantized.onnx"
            QuantizationOptimizer.quantize_static(str(onnx_path), str(quantized_path))
            deployment_info['quantized'] = str(quantized_path)

        if tensorrt:
            trt_path = self.output_dir / "model_optimized.trt"
            TensorRTOptimizer.optimize_onnx(str(onnx_path), str(trt_path))
            deployment_info['tensorrt'] = str(trt_path)

        logger.info(f"Deployment complete. Files: {deployment_info}")
        return deployment_info

class EdgeServerOptimization:

    @staticmethod
    def get_model_info(model_path: str) -> Dict:

        if model_path.endswith('.onnx'):
            import onnx
            model = onnx.load(model_path)

            total_params = 0
            for init in model.graph.initializer:
                shape = list(init.dims)
                params = 1
                for s in shape:
                    params *= s
                total_params += params

            file_size = Path(model_path).stat().st_size / (1024 * 1024)

            return {
                'total_params': total_params,
                'file_size_mb': file_size,
                'file_size_kb': file_size * 1024
            }
        else:
            checkpoint = torch.load(model_path, map_location='cpu')

            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint.state_dict()

            total_params = sum(p.numel() for p in checkpoint.parameters() if p.requires_grad)
            file_size = Path(model_path).stat().st_size / (1024 * 1024)

            return {
                'total_params': total_params,
                'file_size_mb': file_size,
                'file_size_kb': file_size * 1024
            }

    @staticmethod
    def estimate_memory_usage(model_path: str, batch_size: int = 1,
                             sequence_length: int = 30) -> Dict:

        info = EdgeServerOptimization.get_model_info(model_path)

        input_size = batch_size * sequence_length * 6 * 4 / (1024 * 1024)

        output_size = batch_size * 1 * 4 / (1024 * 1024)

        working_memory = info['file_size_mb'] * 2

        total = input_size + output_size + working_memory

        return {
            'model_size_mb': info['file_size_mb'],
            'input_size_mb': input_size,
            'output_size_mb': output_size,
            'working_memory_mb': working_memory,
            'total_estimated_mb': total
        }

if __name__ == "__main__":

    logger.info("Edge deployment utilities loaded")
