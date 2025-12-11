import time
from dataclasses import dataclass
from typing import List, Optional

from flash_embed.core.io.reader import Sample


@dataclass
class Batch:
    samples: List[Sample]


class DynamicBatcher:
    """Batcher with size and timeout triggers."""

    def __init__(self, max_size: int, max_delay_s: float):
        self.max_size = max_size
        self.max_delay_s = max_delay_s
        self._buffer: List[Sample] = []
        self._deadline = time.monotonic() + self.max_delay_s

    def add(self, sample: Sample) -> Optional[Batch]:
        self._buffer.append(sample)
        if len(self._buffer) >= self.max_size:
            return self.flush()
        if time.monotonic() >= self._deadline and self._buffer:
            return self.flush()
        return None

    def flush(self) -> Optional[Batch]:
        if not self._buffer:
            self._deadline = time.monotonic() + self.max_delay_s
            return None
        batch = Batch(samples=list(self._buffer))
        self._buffer.clear()
        self._deadline = time.monotonic() + self.max_delay_s
        return batch
