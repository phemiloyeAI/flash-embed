import asyncio
import argparse
from typing import Any, Dict

from flash_embed.config import load_config
from flash_embed.core.pipeline import AsyncPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flash Embed pipeline runner")
    parser.add_argument("--config", type=str, help="Path to YAML config file", default=None)
    parser.add_argument("--data-path", action="append", help="Data path or shard pattern (repeatable)")
    parser.add_argument("--backend", type=str, help="Model backend (torch|onnx|tensorrt|triton)")
    parser.add_argument("--model-name", type=str, help="Model name/identifier")
    parser.add_argument("--device", type=str, help="Device for model (cuda|cpu)")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument("--triton-url", type=str, help="Triton server URL (host:port)")
    parser.add_argument("--triton-version", type=str, help="Triton model version")
    parser.add_argument("--decode-backend", type=str, help="Decode backend (cpu|dali)")
    return parser.parse_args()


def build_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    if args.data_path:
        overrides.setdefault("io", {})["data_paths"] = args.data_path
    if args.backend:
        overrides.setdefault("model", {})["backend"] = args.backend
    if args.model_name:
        overrides.setdefault("model", {})["name"] = args.model_name
    if args.device:
        overrides.setdefault("model", {})["device"] = args.device
    if args.batch_size:
        overrides.setdefault("batch", {})["size"] = args.batch_size
    if args.output_dir:
        overrides.setdefault("output", {})["out_dir"] = args.output_dir
    if args.triton_url:
        overrides.setdefault("model", {})["triton_url"] = args.triton_url
    if args.triton_version:
        overrides.setdefault("model", {})["triton_version"] = args.triton_version
    if args.decode_backend:
        overrides.setdefault("io", {})["decode_backend"] = args.decode_backend
    return overrides


def main() -> None:
    args = parse_args()
    overrides = build_overrides(args)
    cfg = load_config(args.config, overrides=overrides)
    pipeline = AsyncPipeline(cfg)
    asyncio.run(pipeline.run())
    pipeline.close()


if __name__ == "__main__":
    main()
