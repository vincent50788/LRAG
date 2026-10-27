"""A minimal local RAG system that answers questions about the employee handbook.
sentence-transformers makes the embeddings on this computer. The Gemini API generates the answers.

    python rag.py index                # Offline step: read the documents, split, embed, and store the chunks.
    python rag.py search "question"    # Search only. Shows the chunks that the search finds. Does not call the LLM.
    python rag.py ask "question"       # Full RAG: search, build the prompt, and generate the answer with Gemini.
    python rag.py eval                 # Measure retrieval with eval/questions.json: vector only, and vector + rerank.

Embedding runs on this computer with Qwen3-Embedding from Hugging Face.
Generation uses the Gemini API (free tier). Only the top chunks go to the cloud.
The code is in lrag/. Each file does one step. See lrag/__init__.py.
"""
import sys

from lrag.timing import tick  # Import this first. It starts the timer, so the timer also measures the next import.
from lrag import evaluate, pipeline  # noqa: E402  Loads torch, sentence-transformers, chromadb, and genai.

tick("Import packages (torch, sentence-transformers, chromadb, genai)")


def main():
    cmd, *rest = sys.argv[1:] or ["help"]
    if cmd == "index":
        pipeline.build_index()
    elif cmd == "search" and rest:
        for score, c in pipeline.search(rest[0]):
            print(f"[{score:.3f}] ({c['source']})\n{c['text']}\n")
    elif cmd == "ask" and rest:
        pipeline.ask(rest[0])
    elif cmd == "eval":
        evaluate.run()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
