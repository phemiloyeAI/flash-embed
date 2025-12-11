from typing import Any

from PIL import Image

from .reader import Sample


class Decoder:
    """Minimal decoder; replace with CPU/GPU codecs."""

    def decode(self, sample: Sample) -> Sample:
        if isinstance(sample.image, (bytes, bytearray)):
            img = Image.open(sample.image).convert("RGB")
            return Sample(uid=sample.uid, image=img, text=sample.text, meta=sample.meta)
        return sample
