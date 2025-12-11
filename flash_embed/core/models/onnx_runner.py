from typing import Any, Dict, Sequence

import numpy as np

from flash_embed.core.models.model_runner import ModelRunner


class OnnxRunner(ModelRunner):
    """ONNXRuntime runner with optional CUDA EP and IO binding."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str = "cuda",
        model_path: str | None = None,
        max_batch: int | None = None,
        **_: Any,
    ):
        try:
            import onnxruntime as ort
        except Exception as exc:
            raise RuntimeError("onnxruntime is required for OnnxRunner") from exc

        if not model_path:
            raise ValueError("model_path is required for OnnxRunner")
        
        self.ort = ort
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device != "cpu" else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.device = device
        self._max_batch = max_batch or 64
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        self._preprocess = None  

    def warmup(self) -> None:
        dummy = np.zeros((1, 3, 224, 224), dtype=np.float32)
        feeds = {self.input_names[0]: dummy}
        _ = self.session.run(self.output_names, feeds)

    def max_batch_size(self) -> int:
        return self._max_batch

    def encode(
        self,
        images: Sequence[Any] | None = None,
        texts: Sequence[str] | None = None,
    ) -> Dict[str, np.ndarray]:
        feeds: Dict[str, Any] = {}
        if images is not None:
            if self._preprocess:
                imgs = self._preprocess(images)
            else:
                imgs = np.stack(images)
            feeds[self.input_names[0]] = imgs
       
        outputs = self.session.run(self.output_names, feeds)
        
        return {name: out for name, out in zip(self.output_names, outputs)}

    def close(self) -> None:
        return
