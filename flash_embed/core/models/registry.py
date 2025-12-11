from typing import Dict, Type

from flash_embed.core.models.model_runner import ModelRunner
from flash_embed.core.models.torch_runner import TorchRunner
from flash_embed.core.models.onnx_runner import OnnxRunner
from flash_embed.core.models.tensorrt_runner import TensorRTRunner
from flash_embed.core.models.triton_runner import TritonRunner


REGISTRY: Dict[str, Type[ModelRunner]] = {
    "torch": TorchRunner,
    "onnx": OnnxRunner,
    "tensorrt": TensorRTRunner,
    "triton": TritonRunner,
}


def resolve(backend: str) -> Type[ModelRunner]:
    try:
        return REGISTRY[backend.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported backend: {backend}") from exc
