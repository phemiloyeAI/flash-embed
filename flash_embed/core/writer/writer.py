import json
from pathlib import Path
from typing import Dict

import numpy as np


class Writer:
    """Writer supporting multiple output formats with a simple manifest."""

    def __init__(self, output_dir: str, fmt: str = "npy"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.format = fmt
        self._manifest = []

    def write_batch(self, embeddings: Dict[str, np.ndarray]) -> None:
        """Persist embeddings to disk."""
        for kind, array in embeddings.items():
            filename = self._next_filename(kind)
            if self.format == "npy":
                np.save(filename, array)
            elif self.format == "npz":
                np.savez_compressed(filename, data=array)
            elif self.format in {"parquet", "arrow"}:
                try:
                    import pyarrow as pa
                    import pyarrow.parquet as pq
                except Exception as exc:
                    raise RuntimeError("pyarrow is required for parquet/arrow output") from exc
                table = pa.Table.from_pydict({"embedding": [array.tolist()]})
                pq.write_table(table, filename)
            else:
                raise ValueError(f"Unsupported writer format: {self.format}")
            self._manifest.append({"kind": kind, "path": str(filename)})

    def _next_filename(self, kind: str) -> Path:
        idx = len(self._manifest)
        suffix = self.format if self.format != "arrow" else "arrow"
        return self.output_dir / f"{kind}_{idx}.{suffix}"

    def close(self) -> None:
        manifest_path = self.output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(self._manifest))
