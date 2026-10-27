"""Step 3b, rerank: sort the candidates from the vector search again with a stronger model.

The embedding model is a bi-encoder. It encodes the question and each chunk separately, so the index can store
the chunk vectors before anyone asks. This is fast, but the model never sees the question and the chunk together.
A reranker is a cross-encoder. It reads the question and one chunk together, and gives one relevance score.
It is more accurate, but it needs one model call for each chunk. It cannot use an index.
Thus: the vector search gets CANDIDATES chunks fast. The reranker then reads only those and keeps the best TOP_K.

Qwen3-Reranker is a small LLM. For each pair, it answers "yes" or "no" to an instruction.
The score is P("yes"), from 0 to 1.
The model uses its default instruction: "Given a web search query, retrieve relevant passages that answer the query".
We tried a handbook-specific instruction. With eval/questions.json, it was worse (hit@3 53/54, not 54/54).
"""
from functools import cache

import torch
from sentence_transformers import CrossEncoder

from .config import RERANK_MODEL


@cache
def _model():  # Loads the model (~1.2 GB) only at its first use.
    try:  # Read the local cache first, with no network access (see embedding.py).
        return CrossEncoder(RERANK_MODEL, local_files_only=True)
    except OSError:  # No local copy. Download the model from Hugging Face.
        return CrossEncoder(RERANK_MODEL)


def rerank(question, hits, k):
    """hits: [(similarity, chunk), ...] from the vector search. Returns the best k as [(score, chunk), ...], highest first."""
    chunks = [c for _, c in hits]
    scores = _model().predict([(question, c["text"]) for c in chunks], activation_fn=torch.nn.Sigmoid())
    return sorted(zip(scores.tolist(), chunks), key=lambda x: x[0], reverse=True)[:k]
