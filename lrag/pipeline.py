"""Connects the steps into the RAG flow.

Offline (index)  data/*.md ─split─▶ chunks ─embed─▶ chroma_db/
Online (ask)     question ─embed─▶ top 20 chunks ─rerank─▶ top k chunks ─add to prompt─▶ LLM ─▶ answer
"""
from . import llm, rerank, vector_store
from .chunking import load_chunks
from .config import CANDIDATES, CHROMA_DIR, DATA_DIR, RERANK, TOP_K
from .timing import tick


def build_index():
    chunks = load_chunks(DATA_DIR)
    tick(f"Read and split the files: {len(chunks)} chunks")
    vector_store.build(chunks)
    tick("Store the chunks in the vector store (includes embedding)")
    print(f"Indexed {len(chunks)} chunks → {CHROMA_DIR.name}/")


def search(question, k=TOP_K):
    """Return the k best chunks: [(score, chunk), ...], highest first.
    With rerank, the score is the reranker score. Without rerank, it is the cosine similarity."""
    if not RERANK:
        hits = vector_store.search(question, k)
        tick(f"Search the top {k} chunks (includes index load and question embedding)")
        return hits
    hits = vector_store.search(question, CANDIDATES)
    tick(f"Search the top {CANDIDATES} candidates (includes index load and question embedding)")
    hits = rerank.rerank(question, hits, k)
    tick(f"Rerank the {CANDIDATES} candidates and keep the top {k} (includes reranker load)")
    return hits


def ask(question):
    hits = search(question)
    for score, c in hits:
        print(f"[{score:.3f}] {c['text'].splitlines()[0]}")
    context = "\n\n".join(c["text"] for _, c in hits)
    llm.generate(question, context)
