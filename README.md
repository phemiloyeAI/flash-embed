# 🚀 FlashEmbed

FlashEmbed is a high-throughput, scalable image-embedding pipeline built for datasets containing **millions to billions of images**.  
It provides a clean modular architecture, multiple interchangeable inference backends, and a reliable shard-based orchestration system for distributed processing.

FlashEmbed is designed for:
- Offline embedding of very large datasets  
- High-performance similarity search backends (FAISS)  
- A faster, more scalable alternative to CLIP-retrieval’s embedding pipeline  
- Any workflow requiring efficient large-batch GPU inference at scale  

---

## ✨ Key Features

### 🔌 Modular Backends (Swap Inference Engines Easily)
FlashEmbed supports multiple inference backends with identical APIs:

| Backend | Speed | Setup Difficulty | Description |
|--------|--------|------------------|-------------|
| **Torch** | Medium | Easiest | Zero dependencies. Ideal for development or small jobs. |
| **TensorRT** | Fast | Medium | ONNX → TensorRT optimized engine for maximum single-GPU throughput. |
| **Triton (DALI + TRT)** | Fastest | Hardest | Production-grade dynamic batching + GPU JPEG decode + highest throughput. |

Backends are plug-and-play via a single CLI flag.

---

## 🧱 Architecture Overview

FlashEmbed uses a **shard-first distributed architecture**, enabling linear scaling and fault isolation. The system has four major components:

### 1. Input Shards
- Dataset is stored as WebDataset `.tar` shards (e.g., 10k–50k images each).  
- Works seamlessly with S3, GCS, R2, or local NVMe storage.  
- Avoids overhead from millions of tiny files.

### 2. Orchestrator
A lightweight controller that coordinates shard processing:

- Tracks each shard’s state:  
  `PENDING → IN_PROGRESS → DONE`  
  `IN_PROGRESS → FAILED → PENDING (retry)`
- Ensures no two workers process the same shard simultaneously  
- Handles retries and reclaims stuck shards  
- Supports both **local (SQLite)** and **distributed (SQS/Dynamo)** modes  

### 3. Workers
Each worker performs the following:

1. Acquire next available shard  
2. Download shard to fast local storage  
3. Read images and batch JPEG bytes  
4. Run inference using chosen backend  
5. Write `shard-XXXX.npz` (or Parquet) containing `{id → embedding}`  
6. Upload result and mark shard DONE  

The architecture scales horizontally simply by adding workers.

### 4. Index Builder (FAISS)
After embedding:

- Samples embeddings to train IVF/PQ codebooks  
- Streams all embeddings to build a large-scale FAISS index  
- Produces search-ready index files  

---

## 📦 Project Structure
This structure cleanly isolates:
- backend logic  
- I/O  
- orchestration  
- worker execution  
- indexing 


## 🚀 Quick Start (Torch Backend)

The Torch backend requires no TensorRT or Triton setup.

```bash
pip install flash-embed

flash-embed run-embed \
  --input-manifest=images.parquet \
  --output-dir=embeddings/ \
  --backend=torch \
  --model=openai/clip-vit-base-patch32 \
  --num-workers=4
```

## Roadmap
- Text embeddings + multimodal shard format
- Built-in FAISS search server
- Distributed Triton serving (multi-node)
- More backend engines (ONNX Runtime / DeepSpeed)
- Optional image filtering (NSFW, aesthetic, resolution)