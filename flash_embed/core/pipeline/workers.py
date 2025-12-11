import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from flash_embed.config import Config
from flash_embed.core.io.decoder import Decoder
from flash_embed.core.io.dali_decoder import DaliDecoder
from flash_embed.core.io.reader import Reader, Sample, WebDatasetReader, PrefetchReader
from flash_embed.core.models import resolve, ModelRunner
from flash_embed.core.pipeline.batcher import DynamicBatcher
from flash_embed.core.pipeline.scheduler import Scheduler
from flash_embed.core.telemetry.logging import get_logger
from flash_embed.core.telemetry.metrics import Metrics
from flash_embed.core.writer import Writer


class AsyncPipeline:
    """Async orchestrator: ingest -> decode -> batch -> infer -> write."""

    def __init__(self, config: Config, reader: Optional[Reader] = None):
        self.config = config
        self.logger = get_logger()
        self.metrics = Metrics()

        backend_cls = resolve(config.model.backend)
        self.model_runner: ModelRunner = backend_cls(
            model_name=config.model.name,
            device=config.model.device,
            max_batch=config.model.max_batch,
            model_path=config.model.path,
            triton_url=config.model.triton_url,
            triton_version=config.model.triton_version,
        )

        base_reader: Reader = reader or WebDatasetReader(
            shards=config.io.data_paths,
            decode=config.io.decode,
            shuffle=config.io.shuffle,
        )
        self.reader: Reader = PrefetchReader(base_reader, max_prefetch=config.io.prefetch)

        if config.io.decode_backend == "dali":
            device_id = 0 if config.model.device.startswith("cuda") else -1
            self.decoder = DaliDecoder(device_id=device_id, decode_device=config.io.decode_device)
        else:
            self.decoder = Decoder()

        self.batcher = DynamicBatcher(
            max_size=config.batch.size,
            max_delay_s=config.batch.max_delay_ms / 1000.0,
        )

        self.scheduler = Scheduler()
        self.writer = Writer(config.output.out_dir, fmt=config.output.format)

        cap = config.queues.capacity
        self.raw_q: asyncio.Queue[Optional[Sample]] = asyncio.Queue(maxsize=cap)
        self.decoded_q: asyncio.Queue[Optional[Sample]] = asyncio.Queue(maxsize=cap)
        self.batch_q: asyncio.Queue = asyncio.Queue(maxsize=cap)
        self.output_q: asyncio.Queue = asyncio.Queue(maxsize=cap)

        self.decode_workers = max(1, config.workers.decode_workers)

        self.decode_executor = ThreadPoolExecutor(max_workers=self.decode_workers)
        self.infer_executor = ThreadPoolExecutor(max_workers=config.workers.infer_workers)
        self.writer_executor = ThreadPoolExecutor(max_workers=1)

    async def run(self) -> None:
        tasks = []

        tasks.append(asyncio.create_task(self._read_loop()))

        for _ in range(self.decode_workers):
            tasks.append(asyncio.create_task(self._decode_loop()))

        tasks.append(asyncio.create_task(self._batch_loop()))
        tasks.append(asyncio.create_task(self._infer_loop()))
        tasks.append(asyncio.create_task(self._write_loop()))

        try:
            await asyncio.gather(*tasks)
        finally:
            self.close()

    async def _read_loop(self) -> None:
        iterator = iter(self.reader)
        while True:
            try:
                sample = await asyncio.to_thread(next, iterator)
            except StopIteration:
                break
            except Exception as exc:
                self.logger.error(f"Read failed: {exc}")
                break
            self.scheduler.start(sample.uid)
            await self.raw_q.put(sample)

        for _ in range(self.decode_workers):
            await self.raw_q.put(None)

    async def _decode_loop(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            sample = await self.raw_q.get()
            try:
                if sample is None:
                    await self.decoded_q.put(None)
                    return
                try:
                    decoded = await loop.run_in_executor(
                        self.decode_executor, self.decoder.decode, sample
                    )
                    await self.decoded_q.put(decoded)
                except Exception as exc:
                    self.logger.error(f"Decode failed: {exc}")
                    self.scheduler.fail(sample.uid, str(exc), retry=False)
            finally:
                self.raw_q.task_done()

    async def _batch_loop(self) -> None:
        completed_decoders = 0
        while True:
            sample = await self.decoded_q.get()
            try:
                if sample is None:
                    completed_decoders += 1
                    if completed_decoders == self.decode_workers:
                        batch = self.batcher.flush()
                        if batch:
                            await self.batch_q.put(batch)
                        await self.batch_q.put(None)
                        break
                    continue

                maybe_batch = self.batcher.add(sample)
                if maybe_batch:
                    await self.batch_q.put(maybe_batch)
            finally:
                self.decoded_q.task_done()

    async def _infer_loop(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self.infer_executor, self.model_runner.warmup)

        encode_async = getattr(self.model_runner, "encode_async", None)

        while True:
            batch = await self.batch_q.get()
            try:
                if batch is None:
                    await self.output_q.put(None)
                    break

                images = [s.image for s in batch.samples]
                texts_raw = [s.text for s in batch.samples]

                if all(t is None for t in texts_raw):
                    texts = None
                elif all(t is not None for t in texts_raw):
                    texts = [t for t in texts_raw if t is not None]
                else:
                    self.logger.warning("Mixed presence of text in batch; filling missing as empty.")
                    texts = [t or "" for t in texts_raw]

                try:
                    if callable(encode_async):
                        outputs = await encode_async(images=images, texts=texts)
                    else:
                        outputs = await loop.run_in_executor(
                            self.infer_executor, self.model_runner.encode, images, texts
                        )
                    self.metrics.inc("batches_inferred")
                    for s in batch.samples:
                        self.scheduler.complete(s.uid)
                    await self.output_q.put(outputs)
                except Exception as exc:
                    self.logger.error(f"Inference failed: {exc}")
                    for s in batch.samples:
                        self.scheduler.fail(s.uid, str(exc), retry=False)
            finally:
                self.batch_q.task_done()

    async def _write_loop(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            outputs = await self.output_q.get()
            try:
                if outputs is None:
                    break
                try:
                    await loop.run_in_executor(
                        self.writer_executor, self.writer.write_batch, outputs
                    )
                    self.metrics.inc("batches_written")
                except Exception as exc:
                    self.logger.error(f"Write failed: {exc}")
            finally:
                self.output_q.task_done()

    def close(self) -> None:
        self.reader.close()
        self.writer.close()
        self.model_runner.close()
        self.decode_executor.shutdown(wait=False)
        self.infer_executor.shutdown(wait=False)
        self.writer_executor.shutdown(wait=False)