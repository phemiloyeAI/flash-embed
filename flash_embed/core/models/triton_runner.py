from typing import Any, Dict, Sequence

import numpy as np

from flash_embed.core.models.model_runner import ModelRunner


class TritonRunner(ModelRunner):
    """Triton Inference Server client runner."""

    def __init__(
        self,
        model_name: str,
        max_batch: int | None = None,
        triton_url: str | None = None,
        triton_version: str | None = None,
    ):
        try:
            import tritonclient.grpc as grpcclient
            import tritonclient.grpc.aio as grpcclient_aio
            from tritonclient.utils import InferenceServerException  # noqa: F401
        except Exception as exc:
            raise RuntimeError("tritonclient[grpc] is required for TritonRunner") from exc

        if not triton_url:
            raise ValueError("triton_url is required for TritonRunner")

        self.grpcclient = grpcclient
        self.grpcclient_aio = grpcclient_aio
        self.client = grpcclient.InferenceServerClient(url=triton_url, verbose=False)
        self.async_client = grpcclient_aio.InferenceServerClient(url=triton_url, verbose=False)
        self.model_name = model_name
        self.model_version = triton_version or "1"
        self._max_batch = max_batch or 128

        self.model_metadata = self.client.get_model_metadata(model_name, self.model_version)
        self.model_config = self.client.get_model_config(model_name, self.model_version).config
        self.input_names = [inp.name for inp in self.model_metadata.inputs]
        self.output_names = [out.name for out in self.model_metadata.outputs]

    def warmup(self) -> None:
        # Attempt a lightweight warmup if possible.
        if not self.input_names:
            return
        dummy_shape = self.model_metadata.inputs[0].shape
        dummy = np.zeros(dummy_shape, dtype=np.float32)
        try:
            self.encode(images=[dummy])
        except Exception:
            # Ignore warmup failures; real errors will surface on first request.
            pass

    def max_batch_size(self) -> int:
        return self._max_batch

    def _build_inputs(self, images: Sequence[Any]) -> list:
        np_images = np.asarray(images)
        infer_input = self.grpcclient.InferInput(self.input_names[0], np_images.shape, np_images.dtype.name)
        infer_input.set_data_from_numpy(np_images)
        return [infer_input]

    def _build_outputs(self) -> list:
        return [self.grpcclient.InferRequestedOutput(name) for name in self.output_names]

    def encode(
        self,
        images: Sequence[Any] | None = None,
        texts: Sequence[str] | None = None,
    ) -> Dict[str, np.ndarray]:
        if images is None:
            raise ValueError("TritonRunner expects images input")
        inputs = self._build_inputs(images)
        outputs = self._build_outputs()
        response = self.client.infer(
            model_name=self.model_name,
            model_version=self.model_version,
            inputs=inputs,
            outputs=outputs,
        )
        return {name: response.as_numpy(name) for name in self.output_names}

    async def encode_async(
        self,
        images: Sequence[Any] | None = None,
        texts: Sequence[str] | None = None,
    ) -> Dict[str, np.ndarray]:
        if images is None:
            raise ValueError("TritonRunner expects images input")
        np_images = np.asarray(images)
        infer_input = self.grpcclient_aio.InferInput(self.input_names[0], np_images.shape, np_images.dtype.name)
        infer_input.set_data_from_numpy(np_images)
        outputs = [self.grpcclient_aio.InferRequestedOutput(name) for name in self.output_names]
        response = await self.async_client.infer(
            model_name=self.model_name,
            model_version=self.model_version,
            inputs=[infer_input],
            outputs=outputs,
        )
        return {name: response.as_numpy(name) for name in self.output_names}

    def close(self) -> None:
        return
