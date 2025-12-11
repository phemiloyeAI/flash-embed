from typing import Any, Dict, Protocol, Sequence

import numpy as np


class ModelRunner(Protocol):
    """Unified model API across backends."""

    def warmup(self) -> None:
        ...

    def max_batch_size(self) -> int:
        ...

    def encode(
        self,
        images: Sequence[Any] | None = None,
        texts: Sequence[str] | None = None,
    ) -> Dict[str, np.ndarray]:
        ...

    def close(self) -> None:
        ...
