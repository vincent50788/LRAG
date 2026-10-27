"""All settings are in this file. Environment variables can override the model names."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"  # The documents to search (*.md).
CHROMA_DIR = ROOT / "chroma_db"  # The vector index. `rag.py index` makes it.
COLLECTION = "handbook"

EMBED_MODEL = os.getenv("EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")  # If you change the model, run index again.
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-flash-latest")  # Always the latest Flash model. The free tier allows it.
FALLBACK_MODEL = "gemini-flash-lite-latest"  # The free tier is often busy. If the main model fails, use this lighter model.

TOP_K = 3  # The number of chunks to put in the prompt.

RERANK_MODEL = os.getenv("RERANK_MODEL", "Qwen/Qwen3-Reranker-0.6B")
RERANK = os.getenv("RERANK", "1") != "0"  # RERANK=0 turns off rerank. Search then uses only the vector order.
CANDIDATES = 20  # With rerank: the vector search gets this many chunks. The reranker keeps the best TOP_K.
