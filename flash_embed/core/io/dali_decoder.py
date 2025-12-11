from typing import Any

import numpy as np

from flash_embed.core.io.reader import Sample


class DaliDecoder:
    """Minimal DALI-based decoder (bytes -> RGB uint8)."""

    def __init__(self, device_id: int = 0, decode_device: str = "mixed", batch_size: int = 1):
        try:
            from nvidia.dali.pipeline import Pipeline
            import nvidia.dali.fn as fn
            import nvidia.dali.types as types
        except Exception as exc:
            raise RuntimeError("nvidia-dali is required for DaliDecoder") from exc

        self.fn = fn
        self.types = types

        class _Pipe(Pipeline):
            def __init__(self, batch_size: int, num_threads: int, device_id: int, decode_device: str):
                super().__init__(batch_size=batch_size, num_threads=num_threads, device_id=device_id, seed=12)
                self.decode_device = decode_device
                self._input = fn.external_source(name="images", device="cpu")
                output = fn.decoders.image(
                    self._input,
                    device=self.decode_device,
                    output_type=types.RGB,
                )
                self.output = output
                self._data = None
                self.build()

            def iter_setup(self):
                if self._data is None:
                    raise RuntimeError("No data fed to DALI pipeline")
                self.feed_input("images", self._data)

            def feed(self, data: Any):
                self._data = data

        self._pipe = _Pipe(batch_size=batch_size, num_threads=2, device_id=device_id, decode_device=decode_device)

    def decode(self, sample: Sample) -> Sample:
        if not isinstance(sample.image, (bytes, bytearray)):
            return sample
        self._pipe.feed([sample.image])
        outputs = self._pipe.run()
        decoded = outputs[0][0]
        decoded_cpu = decoded.as_cpu().as_array()
        return Sample(uid=sample.uid, image=decoded_cpu, text=sample.text, meta=sample.meta)
