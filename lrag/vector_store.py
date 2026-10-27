"""Step 3, storage: the Chroma vector database. It runs locally and keeps its data in chroma_db/.

When Chroma makes the index, it builds an HNSW graph. A search visits only a small part of the graph.
Thus a search stays fast, also with many chunks.
The cost: the results are approximate. Sometimes a search misses the true first result.
Chroma also adds one dependency and one more folder.

Chroma calculates the vectors itself. This module gives Chroma only text (see embedding.py):
- In add(), Chroma encodes the chunks with Qwen3EF.__call__.
- In query(query_texts=...), Chroma encodes the question with Qwen3EF.embed_query.
"""
import shutil
from functools import cache

import chromadb
from chromadb.api.client import SharedSystemClient
from chromadb.config import Settings

from .config import CHROMA_DIR, COLLECTION
from .embedding import Qwen3EF


@cache
def _client():
    return chromadb.PersistentClient(CHROMA_DIR, Settings(anonymized_telemetry=False))  # Do not send usage statistics.


@cache
def _ef():
    return Qwen3EF()


def build(chunks):
    """chunks: [{"source", "text"}, ...]. Rebuild the full index. First, delete the old index folder."""
    # Do not use delete_collection(). It deletes the HNSW folder only if this process loaded that collection.
    # A new `rag.py index` process does not load it. Thus each rebuild leaves one more unused folder.
    _client.cache_clear()  # Close the open clients before rmtree deletes the folder.
    SharedSystemClient.clear_system_cache()
    shutil.rmtree(CHROMA_DIR, ignore_errors=True)
    col = _client().create_collection(
        COLLECTION,
        configuration={"hnsw": {"space": "cosine"}},  # The default is L2 distance. Use cosine similarity.
        embedding_function=_ef(),  # Chroma does not save a custom EF. Give it on each get_collection call.
    )
    # ponytail: One add() call adds all chunks.
    # Above client.get_max_batch_size() (about 5000 chunks), add the chunks in batches.
    col.add(
        ids=[str(i) for i in range(len(chunks))],
        documents=[c["text"] for c in chunks],
        metadatas=[{"source": c["source"]} for c in chunks],
    )


def search(q, k):
    """q is the question text. Chroma encodes it with embed_query. Returns [(similarity, chunk), ...], highest first."""
    col = _client().get_collection(COLLECTION, embedding_function=_ef())
    r = col.query(query_texts=[q], n_results=k)
    return [
        (1 - d, {"source": m["source"], "text": t})  # Chroma returns a distance. Distance = 1 − cosine similarity.
        for d, m, t in zip(r["distances"][0], r["metadatas"][0], r["documents"][0])
    ]
