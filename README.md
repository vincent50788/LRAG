# LRAG: a local RAG system for an employee handbook (for learning)

- **Search** runs on this computer: [sentence-transformers](https://sbert.net) with `Qwen/Qwen3-Embedding-0.6B`. The handbook does not leave the computer.
- **Generation** runs in the cloud: the Gemini API free tier. The system sends only the question and the top 3 chunks.
  (⚠️ Google can use free-tier content to improve its products. For real company documents, use a paid tier or a local model.)

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
export GEMINI_API_KEY=...                # Get a free key at aistudio.google.com.
.venv/bin/python rag.py index            # The first run downloads the embedding model (about 1.1 GB).
.venv/bin/python rag.py ask "What do I do if I forget to clock in?"
```

The handbook in `data/` is for a fictional company, Starriver Tech.

## What RAG does

An LLM did not see your company handbook during training. RAG (Retrieval-Augmented Generation) **finds the related text first, then puts that text in the prompt**. The LLM then answers from that text.

A **chunk** is one part of a document. The system stores, finds, and sends each chunk as one unit.

```
Offline (index)  data/*.md ─split─▶ chunks ─embed─▶ vectors ─▶ chroma_db/
Online (ask)     question ─embed─▶ vector ─cosine similarity─▶ top 3 chunks ─add to prompt─▶ LLM ─▶ answer
```

| Step         | Location                            | Key points                                                                                                                                                                |
| ------------ | ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Chunking     | `lrag/chunking.py` `split()`        | Splits the text at each `##` heading. If a chunk is too large, unrelated text dilutes its vector. If a chunk is too small, it loses its context.                          |
| Embedding    | `lrag/embedding.py` `Qwen3EF`       | Sentences with a similar meaning get vectors with a similar direction. Use the **same** model for questions and chunks. If you change the model, run `index` again. Qwen3 needs the `query` prompt for questions, not for chunks. |
| Retrieval    | `lrag/vector_store.py` `search()`   | Sorts the chunks by cosine similarity and takes the top K. See the Chroma section below.                                                                                  |
| Augmentation | `lrag/llm.py` `SYSTEM_PROMPT`       | Gives the model three rules: use only the reference material, give the source, and say "I do not know" when the answer is missing. These rules reduce hallucination. |
| Generation   | `lrag/llm.py` `generate()`          | Calls the Gemini API and prints the answer as a stream. Retrieval and generation can use models from different vendors.                                                    |

## Project structure

```
rag.py              Command-line entry point: index / search / ask
lrag/
  config.py         Settings: paths, model names, Top-K (environment variables can override the models)
  timing.py         tick(): prints the time of each step
  chunking.py       Step 1: chunking
  embedding.py      Step 2: embedding (Qwen3EF)
  vector_store.py   Step 3: Chroma index and search
  llm.py            Step 4: prompt and Gemini generation
  pipeline.py       Connects steps 1–4 into build_index() / search() / ask()
data/               The documents to search (*.md)
chroma_db/          The vector index (`rag.py index` makes it, git ignores it)
```

## Experiments

1. Run `.venv/bin/python rag.py search "Can I use my own phone for work?"`.
   The first result is "Remote Work", not "Device Use".
   The question and the Remote Work chunk both talk about work outside the office.
   The Device Use chunk contains the answer (BYOD), but it also contains other topics, such as laptop encryption and software.
   The correct chunk is in position 2. Top-3 retrieval still finds it. **This is why K is not 1.**
2. Change `split()` in `lrag/chunking.py` to split at each line. Run `index` again. Then search the same question again and compare the scores.
3. Run `ask "Does the company give parking spaces to employees?"`. The handbook does not answer this. Check if the model says that it does not know.
4. Remove the rules from `SYSTEM_PROMPT` in `lrag/llm.py`. Ask question 3 again and look for hallucination.
5. Add a new .md file to `data/` and run `index` again. You can then ask about the new content with no code change.
6. Compare embedding models: run `EMBED_MODEL=Qwen/Qwen3-Embedding-4B .venv/bin/python rag.py index`, then run `search`.
   **Do not compare scores from different models directly.** Compare the gap between result 1 and result 2. A larger gap shows that the model separates the correct chunk better.
7. Search with a question in a different language, for example Spanish. A multilingual embedding model can search across languages.

## Vector store: Chroma (`lrag/vector_store.py`)

Interface: `build(chunks)` stores the chunks. `search(q, k)` returns the top k as `[(similarity, chunk), ...]`. The index is in `chroma_db/`. If you change the embedding model, run `index` again.

- The code gives Chroma only text. The custom `Qwen3EF` (in `lrag/embedding.py`) calculates the vectors. Chroma calls it as its `embedding_function`:
  - For chunks, Chroma calls `__call__`.
  - For questions, Chroma calls `embed_query`, which adds the `query` prefix.
- Chroma uses an HNSW graph index. A search visits only part of the graph, so it stays fast with many chunks. But the results are **approximate**. Sometimes a search misses the true first result.
- Chroma supports persistent storage and metadata filters (for example, search only one document).

## Not included on purpose (possible later additions)

- Hybrid search (BM25 keywords and vectors): vector search often fails on proper nouns and ID numbers.
- Rerank: get 20 candidates first, then use a reranker model to select the best 3.
- PDF and Word parsing, multi-turn conversation, a web UI.
- A local LLM for generation (for example, `ollama pull qwen3:8b`): generation also runs offline, and no data leaves the company.
