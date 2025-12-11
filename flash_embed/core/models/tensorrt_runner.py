from typing import Any, Dict, Sequence

import numpy as np

from flash_embed.core.models.model_runner import ModelRunner


class TensorRTRunner(ModelRunner):
    """TensorRT runner stub; expects pre-built engine."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        max_batch: int | None = None,
        **_: Any,
    ):
        try:
            import tensorrt as trt 
            import pycuda.driver as cuda  # type: ignore
            import pycuda.autoinit  # noqa: F401
        except Exception as exc:
            raise RuntimeError("tensorrt and pycuda are required for TensorRTRunner") from exc

        self.trt = trt
        self.cuda = cuda
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(model_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.device = device
        self._max_batch = max_batch or self.engine.max_batch_size
        self.bindings = []
        self.inputs = []
        self.outputs = []
        self.allocations = []
        for binding in self.engine:
            binding_idx = self.engine.get_binding_index(binding)
            shape = self.engine.get_binding_shape(binding_idx)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            size = int(np.prod(shape)) * dtype().itemsize
            device_mem = cuda.mem_alloc(size)
            self.bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                self.inputs.append((binding, shape, dtype, device_mem))
            else:
                self.outputs.append((binding, shape, dtype, device_mem))
            self.allocations.append(device_mem)

    def warmup(self) -> None:
        if not self.inputs:
            return
        name, shape, dtype, device_mem = self.inputs[0]
        host = np.zeros(shape, dtype=dtype)
        self.cuda.memcpy_htod(device_mem, host)
        self.context.execute_v2(self.bindings)

    def max_batch_size(self) -> int:
        return int(self._max_batch)

    def encode(
        self,
        images: Sequence[Any] | None = None,
        texts: Sequence[str] | None = None,
    ) -> Dict[str, np.ndarray]:
        # Minimal implementation assumes single image input and single output.
        if images is None:
            raise ValueError("TensorRTRunner expects images input")
        np_input = np.asarray(images)
        # TODO: apply preprocessing externally before calling encode.
        name, shape, dtype, device_mem = self.inputs[0]
        if np_input.size != np.prod(shape):
            np_input = np_input.reshape(shape)
        self.cuda.memcpy_htod(device_mem, np_input.astype(dtype))
        self.context.execute_v2(self.bindings)
        outputs: Dict[str, np.ndarray] = {}
        for name, shape, dtype, device_mem in self.outputs:
            host_out = np.empty(shape, dtype=dtype)
            self.cuda.memcpy_dtoh(host_out, device_mem)
            outputs[name] = host_out
        return outputs

    def close(self) -> None:
        for alloc in self.allocations:
            alloc.free()
