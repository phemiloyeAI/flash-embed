import asyncio

from flash_embed.config import Config
from flash_embed.core.pipeline import AsyncPipeline


def main() -> None:
    cfg = Config()
    pipeline = AsyncPipeline(cfg)
    asyncio.run(pipeline.run())
    pipeline.close()


if __name__ == "__main__":
    main()
