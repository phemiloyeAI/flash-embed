from dataclasses import dataclass, field, is_dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ModelConfig:
    backend: str = "torch"  
    name: str = "ViT-B/32"
    path: Optional[str] = None
    device: str = "cuda"
    max_batch: Optional[int] = None


@dataclass
class IOConfig:
    data_paths: List[str] = field(default_factory=list)  # webdataset shards or directories
    decode: str = "pil"
    decode_backend: str = "cpu"
    decode_device: str = "gpu"  
    shuffle: bool = False
    prefetch: int = 2


@dataclass
class BatchConfig:
    size: int = 32
    max_delay_ms: int = 10


@dataclass
class QueueConfig:
    capacity: int = 512


@dataclass
class WorkerConfig:
    reader_threads: int = 2
    decode_workers: int = 2
    infer_workers: int = 1  # per GPU process


@dataclass
class RetryConfig:
    max_retries: int = 3
    backoff_ms: int = 100


@dataclass
class OutputConfig:
    out_dir: str = "outputs"
    format: str = "npy"  # parquet | arrow | npz | zarr


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    io: IOConfig = field(default_factory=IOConfig)
    batch: BatchConfig = field(default_factory=BatchConfig)
    queues: QueueConfig = field(default_factory=QueueConfig)
    workers: WorkerConfig = field(default_factory=WorkerConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def _update_dataclass(obj: Any, updates: Dict[str, Any]) -> None:
    """Recursively apply updates to a dataclass instance."""
    for key, value in updates.items():
        if not hasattr(obj, key):
            continue
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            _update_dataclass(current, value)
        elif value is not None:
            setattr(obj, key, value)


def load_config(path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> Config:
    """Load config from YAML and apply overrides."""
    cfg = Config()
    if path:
        try:
            import yaml
        except Exception as exc:
            raise RuntimeError("pyyaml is required to load config files") from exc
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        if isinstance(data, dict):
            _update_dataclass(cfg, data)
    if overrides:
        _update_dataclass(cfg, overrides)
    return cfg
