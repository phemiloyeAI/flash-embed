import os
import random
import threading
import queue
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Protocol

import webdataset as wds
from braceexpand import braceexpand


@dataclass
class Sample:
    uid: str
    image: Any
    text: Optional[str]
    meta: Dict[str, Any]


class Reader(Protocol):
    def __iter__(self) -> Iterator[Sample]:
        ...

    def close(self) -> None:
        ...


class WebDatasetReader:
    """Streaming reader for WebDataset shards."""

    def __init__(self, shards: Iterable[str], decode: str = "pil", shuffle: bool = False):
        self.shards = list(shards)
        self.decode = decode
        self.shuffle = shuffle
        self._pipeline = None

    def __iter__(self) -> Iterator[Sample]:
        if not self.shards:
            return iter([])
        shards = [s for pattern in list(self.shards) for s in braceexpand(pattern)]
        if self.shuffle:
            random.shuffle(shards)

        dataset = wds.WebDataset(shards, handler=wds.handlers.warn_and_continue)
        # keep image bytes for decoding later (e.g with DALI)
        if self.decode:
            dataset = dataset.decode(self.decode)
        if self.shuffle:
            dataset = dataset.shuffle(1000)
        dataset = dataset.to_tuple("jpg", "txt", "__key__")
        for img, txt, key in dataset:
            yield Sample(uid=key, image=img, text=txt, meta={})

    def close(self) -> None:
        return


def iter_images_from_directory(directory_path: str) -> Iterator[Sample]:
    """Fallback reader for plain directories; yields Sample objects."""
    from PIL import Image

    if not os.path.isdir(directory_path):
        raise NotADirectoryError(f"The specified path is not a directory: {directory_path}")

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            try:
                image = Image.open(file_path).convert("RGB")
                yield Sample(uid=filename, image=image, text=None, meta={})
            except Exception as exc:
                raise RuntimeError(f"Failed to read image file {file_path}: {exc}") from exc


class PrefetchReader:
    """Wraps a Reader and prefetches samples using a background thread."""

    def __init__(self, reader: Reader, max_prefetch: int = 2):
        self.reader = reader
        self.max_prefetch = max_prefetch
        self._queue: queue.Queue[Optional[Sample]] = queue.Queue(maxsize=max_prefetch)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._started = False

    def _run(self) -> None:
        try:
            for sample in self.reader:
                self._queue.put(sample)
        finally:
            self._queue.put(None)

    def __iter__(self) -> Iterator[Sample]:
        if not self._started:
            self._thread.start()
            self._started = True
        while True:
            sample = self._queue.get()
            if sample is None:
                break
            yield sample

    def close(self) -> None:
        self.reader.close()
