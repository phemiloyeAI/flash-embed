from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple
import os
import time
import concurrent.futures as cf

import webdataset


@dataclass(frozen=True)
class Record:
    path: Path
    relpath: str
    key: str
    label: Optional[str] = None


@dataclass
class Metrics:
    start_time: float = 0.0
    end_time: float = 0.0

    wall_total: float = 0.0
    wall_pipeline: float = 0.0
    wall_read: float = 0.0
    wall_write: float = 0.0

    last_log_time: float = 0.0
    num_samples: int = 0
    num_bytes: int = 0

    t_enum: float = 0.0
    t_read: float = 0.0
    t_write: float = 0.0

    sps_total: float = 0.0
    sps_read: float = 0.0
    sps_write: float = 0.0


def start_timer() -> float:
    return time.perf_counter()


def stop_timer(t0: float) -> float:
    return time.perf_counter() - t0


def _is_image(p: Path) -> bool:
    s = p.suffix.lower()
    return s in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def make_key_from_relpath(relpath: str) -> str:
    rp = relpath.replace("\\", "/")
    if "." in rp:
        rp = rp.rsplit(".", 1)[0]
    return rp


def iter_train_records(train_root: Path) -> Iterator[Record]:
    for wnid_dir in sorted([p for p in train_root.iterdir() if p.is_dir()]):
        images_dir = wnid_dir / "images"
        if not images_dir.exists():
            continue
        for img_path in sorted([p for p in images_dir.iterdir() if p.is_file() and _is_image(p)]):
            relpath = f"{wnid_dir.name}/images/{img_path.name}"
            key = make_key_from_relpath(relpath)
            yield Record(path=img_path, relpath=relpath, key=key, label=wnid_dir.name)


def iter_test_records(split_root: Path) -> Iterator[Record]:
    images_dir = split_root / "images"
    for img_path in sorted([p for p in images_dir.iterdir() if p.is_file() and _is_image(p)]):
        relpath = f"images/{img_path.name}"
        key = make_key_from_relpath(relpath)
        yield Record(path=img_path, relpath=relpath, key=key, label=None)


def iter_split_records(dataset_root: Path, split: str) -> Iterator[Record]:
    if split == "train":
        yield from iter_train_records(dataset_root / "train")
    elif split in ["test", "val"]:
        yield from iter_test_records(dataset_root / split)
    else:
        raise ValueError(f"unknown split: {split}")


def collect_split_records(dataset_root: Path, split: str):
    t0 = start_timer()
    records = list(iter_split_records(dataset_root, split))
    return records, stop_timer(t0)


def read_image_bytes(record: Record) -> Tuple[Record, bytes, float, float, float]:
    t0 = time.perf_counter()
    data = record.path.read_bytes()
    t1 = time.perf_counter()
    return record, data, t1 - t0, t0, t1


def make_wds_sample(record: Record, img_bytes: bytes) -> dict:
    suffix = record.path.suffix.lower()
    ext = "png" if suffix == ".png" else "jpg"
    return {
        "__key__": record.key,
        ext: img_bytes,
        "path.txt": record.relpath.encode("utf-8"),
    }


def maybe_log_progress(metrics: Metrics, every_secs: float = 2.0) -> None:
    now = time.perf_counter()
    if metrics.last_log_time == 0.0:
        metrics.last_log_time = now
        return
    if now - metrics.last_log_time < every_secs:
        return
    elapsed = now - metrics.start_time
    if elapsed <= 0:
        return
    sps = metrics.num_samples / elapsed
    mb = metrics.num_bytes / (1024 * 1024)
    mbps = mb / elapsed
    rps = metrics.num_samples / metrics.wall_read if metrics.wall_read > 0 else 0.0
    wps = metrics.num_samples / metrics.wall_write if metrics.wall_write > 0 else 0.0
    print(
        f"samples={metrics.num_samples}  "
        f"elapsed={elapsed:.1f}s  "
        f"read_MB={mb:.1f}  "
        f"MBps={mbps:.1f}  "
        f"read_sps={rps:.1f}  "
        f"write_sps={wps:.1f} "
        f"throughput={sps:.1f} samples/s"
    )
    metrics.last_log_time = now


def write_split_shards(
    dataset_root: Path,
    split: str,
    out_pattern: str,
    shard_size: int,
    num_reader_threads: int,
    max_inflight: int,
) -> Metrics:
    total_t0 = time.perf_counter()
    metrics = Metrics()

    records, t_enum = collect_split_records(dataset_root, split)
    metrics.t_enum = t_enum
    metrics.start_time = time.perf_counter()

    out_dir = Path(out_pattern).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    sink = webdataset.ShardWriter(out_pattern, maxcount=shard_size)

    inflight = set()
    read_span_start: Optional[float] = None
    read_span_end: Optional[float] = None

    try:
        with cf.ThreadPoolExecutor(max_workers=num_reader_threads) as ex:
            it = iter(records)

            def submit_one():
                rec = next(it)
                fut = ex.submit(read_image_bytes, rec)
                inflight.add(fut)

            # prefill the inflight set with some tasks
            for _ in range(min(max_inflight, len(records))):
                try:
                    submit_one()
                except StopIteration:
                    break

            while inflight:
                done, _ = cf.wait(inflight, return_when=cf.ALL_COMPLETED)
                for fut in done:
                    inflight.remove(fut)

                    record, data, t_read, tr0, tr1 = fut.result()
                    metrics.t_read += t_read
                    metrics.num_bytes += len(data)
                    if read_span_start is None or tr0 < read_span_start:
                        read_span_start = tr0
                    if read_span_end is None or tr1 > read_span_end:
                        read_span_end = tr1
                    if read_span_start is not None and read_span_end is not None:
                        metrics.wall_read = max(0.0, read_span_end - read_span_start)

                    sample = make_wds_sample(record, data)
                    tw0 = time.perf_counter()
                    sink.write(sample)
                    tw1 = time.perf_counter()
                    metrics.t_write += tw1 - tw0
                    metrics.wall_write = metrics.t_write

                    metrics.num_samples += 1
                    maybe_log_progress(metrics)

                    try:
                        submit_one()
                    except StopIteration:
                        pass

    finally:
        sink.close()

    metrics.end_time = time.perf_counter()
    metrics.wall_total = metrics.end_time - total_t0
    metrics.wall_pipeline = metrics.end_time - metrics.start_time
    metrics.sps_total = metrics.num_samples / metrics.wall_pipeline if metrics.wall_pipeline > 0 else 0.0
    metrics.sps_read = metrics.num_samples / metrics.wall_read if metrics.wall_read > 0 else 0.0
    metrics.sps_write = metrics.num_samples / metrics.wall_write if metrics.wall_write > 0 else 0.0

    return metrics


def write_all_splits(
    dataset_root: Path,
    out_dir: Path,
    shard_size: int = 1000,
    num_reader_threads: int = 16,
    max_inflight: int = 256,
) -> Dict[str, Metrics]:
    out_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Metrics] = {}
    for split in ("train", "val", "test"):
        out_pattern = str(out_dir / split / f"{split}-%06d.tar")
        results[split] = write_split_shards(
            dataset_root=dataset_root,
            split=split,
            out_pattern=out_pattern,
            shard_size=shard_size,
            num_reader_threads=num_reader_threads,
            max_inflight=max_inflight,
        )
    return results


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--shard-size", type=int, default=15000)
    parser.add_argument("--num-reader-threads", type=int, default=32)
    parser.add_argument("--max-inflight", type=int, default=500)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    shard_size = args.shard_size
    num_reader_threads = min(os.cpu_count() or 1, args.num_reader_threads)
    max_inflight = args.max_inflight

    print(f"Using {num_reader_threads} reader threads")

    results = write_all_splits(
        dataset_root=dataset_root,
        out_dir=output_dir,
        shard_size=shard_size,
        num_reader_threads=num_reader_threads,
        max_inflight=max_inflight,
    )

    with open(output_dir / "metrics.json", "w") as f:
        json.dump({split: metrics.__dict__ for split, metrics in results.items()}, f, indent=2)
