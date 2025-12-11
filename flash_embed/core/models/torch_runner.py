from typing import Any, Dict, Sequence, Union

import numpy as np
import torch
from all_clip import load_clip

from flash_embed.core.models.model_runner import ModelRunner


class TorchRunner(ModelRunner):
    """Torch implementation of ModelRunner."""

    def __init__(
        self,
        model_name: str = "ViT-B/32",
        device: Union[str, torch.device] = "cuda",
        max_batch: int | None = None,
        model_path: str | None = None,  # unused for torch
        **_: Any,
    ):
        self.device = torch.device(device)
        self.model, self.preprocess, self.tokenizer = load_clip(model_name, device=self.device)
        self.model.eval()
        self._max_batch = max_batch

    def warmup(self) -> None:
        with torch.no_grad():
            dummy_img = torch.zeros(1, 3, 224, 224, device=self.device)
            _ = self.model.encode_image(dummy_img)
            dummy_txt = self.tokenizer(["warmup"]).to(self.device)
            _ = self.model.encode_text(dummy_txt)

    def max_batch_size(self) -> int:
        return self._max_batch or 64

    def _prep_texts(self, texts: Sequence[str]) -> torch.Tensor:
        return self.tokenizer(list(texts)).to(self.device)

    def _prep_images(self, images: Sequence[Any]) -> torch.Tensor:
        preprocessed_images = [self.preprocess(image).unsqueeze(0) for image in images]
        return torch.cat(preprocessed_images, dim=0).to(self.device, non_blocking=True)

    def encode(
        self,
        images: Sequence[Any] | None = None,
        texts: Sequence[str] | None = None,
    ) -> Dict[str, np.ndarray]:
        outputs: Dict[str, np.ndarray] = {}
        with torch.no_grad():
            if images:
                batch_images = self._prep_images(images)
                outputs["image"] = self.model.encode_image(batch_images).cpu().numpy()
            if texts:
                tokenized = self._prep_texts(texts)
                outputs["text"] = self.model.encode_text(tokenized).cpu().numpy()
        return outputs

    def close(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
