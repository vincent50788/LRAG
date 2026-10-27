# LRAG: a local RAG system for an employee handbook (for learning)

- **Search** runs on this computer: [sentence-transformers](https://sbert.net) with `Qwen/Qwen3-Embedding-0.6B`, then `Qwen/Qwen3-Reranker-0.6B` sorts the results again. The handbook does not leave the computer.
- **Generation** runs in the cloud: the Gemini API free tier. The system sends only the question and the top 3 chunks.
  (⚠️ Google can use free-tier content to improve its products. For real company documents, use a paid tier or a local model.)

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
export GEMINI_API_KEY=...                # Get a free key at aistudio.google.com.
.venv/bin/python rag.py index            # The first run downloads the embedding model (about 1.1 GB).
.venv/bin/python rag.py ask "What do I do if I forget to clock in?"   # The first run downloads the reranker (about 1.2 GB).
.venv/bin/python rag.py eval             # Measure retrieval with and without rerank.
```

The 12 documents in `data/` (127 chunks) are for a fictional company, Starriver Tech.
The models go to the Hugging Face cache, `~/.cache/huggingface/hub/`, not to this folder.

## What RAG does

An LLM did not see your company handbook during training. RAG (Retrieval-Augmented Generation) **finds the related text first, then puts that text in the prompt**. The LLM then answers from that text.

A **chunk** is one part of a document. The system stores, finds, and sends each chunk as one unit.

```
Offline (index)  data/*.md ─split─▶ chunks ─embed─▶ vectors ─▶ chroma_db/
Online (ask)     question ─embed─▶ vector ─cosine similarity─▶ top 20 chunks ─rerank─▶ top 3 chunks ─add to prompt─▶ LLM ─▶ answer
```

| Step         | Location                            | Key points                                                                                                                                                                |
| ------------ | ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Chunking     | `lrag/chunking.py` `split()`        | Splits the text at each `##` heading. If a chunk is too large, unrelated text dilutes its vector. If a chunk is too small, it loses its context.                          |
| Embedding    | `lrag/embedding.py` `Qwen3EF`       | Sentences with a similar meaning get vectors with a similar direction. Use the **same** model for questions and chunks. If you change the model, run `index` again. Qwen3 needs the `query` prompt for questions, not for chunks. |
| Retrieval    | `lrag/vector_store.py` `search()`   | Sorts the chunks by cosine similarity and takes the top 20 candidates. See the Chroma section below.                                                                      |
| Rerank       | `lrag/rerank.py` `rerank()`         | A cross-encoder reads the question and each candidate together, and keeps the best 3. More accurate than vectors, but slower. See the Rerank section below. `RERANK=0` turns it off. |
| Augmentation | `lrag/llm.py` `SYSTEM_PROMPT`       | Gives the model three rules: use only the reference material, give the source, and say "I do not know" when the answer is missing. These rules reduce hallucination. |
| Generation   | `lrag/llm.py` `generate()`          | Calls the Gemini API and prints the answer as a stream. Retrieval and generation can use models from different vendors.                                                    |

## Project structure

```
rag.py              Command-line entry point: index / search / ask / eval
lrag/
  config.py         Settings: paths, model names, Top-K, rerank (environment variables can override the models)
  timing.py         tick(): prints the time of each step
  chunking.py       Step 1: chunking
  embedding.py      Step 2: embedding (Qwen3EF)
  vector_store.py   Step 3: Chroma index and search
  rerank.py         Step 3b: rerank the candidates with Qwen3-Reranker
  llm.py            Step 4: prompt and Gemini generation
  pipeline.py       Connects steps 1–4 into build_index() / search() / ask()
  evaluate.py       `rag.py eval`: hit@1 and hit@3, with and without rerank
data/               The documents to search (*.md)
eval/questions.json Test questions and the correct chunk for each one
chroma_db/          The vector index (`rag.py index` makes it, git ignores it)
```

## Experiments

1. Run `RERANK=0 .venv/bin/python rag.py search "Can I use my own phone for work?"`.
   "Device Use" is not the first result.
   Other chunks also talk about phones, accounts, and work outside the office.
   The Device Use chunk contains the answer (BYOD), but it also contains other topics, such as laptop encryption and software.
   The correct chunk is in position 3. Top-3 retrieval still finds it. **This is why K is not 1.**
   Then run the same search without `RERANK=0`. The reranker moves Device Use to position 1.
2. Change `split()` in `lrag/chunking.py` to split at each line. Run `index` again. Then search the same question again and compare the scores.
3. Run `ask "Does the company give parking spaces to employees?"`. The handbook does not answer this. Check if the model says that it does not know.
4. Remove the rules from `SYSTEM_PROMPT` in `lrag/llm.py`. Ask question 3 again and look for hallucination.
5. Add a new .md file to `data/` and run `index` again. You can then ask about the new content with no code change.
6. Compare embedding models: run `EMBED_MODEL=Qwen/Qwen3-Embedding-4B .venv/bin/python rag.py index`, then run `search`.
   **Do not compare scores from different models directly.** Compare the gap between result 1 and result 2. A larger gap shows that the model separates the correct chunk better.
7. Search with a question in a different language, for example Spanish. A multilingual embedding model can search across languages.
8. Run `rag.py eval`. Look at the questions that go down with rerank, not only the ones that go up.
   Then add your own questions to `eval/questions.json`. Without a test set, you cannot know if a change helps.

## Vector store: Chroma (`lrag/vector_store.py`)

Interface: `build(chunks)` stores the chunks. `search(q, k)` returns the top k as `[(similarity, chunk), ...]`. The index is in `chroma_db/`. If you change the embedding model, run `index` again.

- The code gives Chroma only text. The custom `Qwen3EF` (in `lrag/embedding.py`) calculates the vectors. Chroma calls it as its `embedding_function`:
  - For chunks, Chroma calls `__call__`.
  - For questions, Chroma calls `embed_query`, which adds the `query` prefix.
- Chroma uses an HNSW graph index. A search visits only part of the graph, so it stays fast with many chunks. But the results are **approximate**. Sometimes a search misses the true first result.
- Chroma supports persistent storage and metadata filters (for example, search only one document).

## Rerank (`lrag/rerank.py`)

- The embedding model is a **bi-encoder**. It encodes the question and each chunk separately. The index can store the chunk vectors before anyone asks, so search is fast. But the model never sees the question and the chunk together.
- The reranker is a **cross-encoder**. It reads the question and one chunk together and gives one score. It is more accurate, but it needs one model call for each chunk, and it cannot use an index.
- Thus the two work together: the vector search gets 20 candidates fast. The reranker reads only those 20 and keeps the best 3.
- Rerank can only change the order. If the correct chunk is not in the 20 candidates, rerank cannot find it.
- Do not compare the two scores. With rerank, `search` shows P("yes") from the reranker (0–1), not the cosine similarity.

Results with `rag.py eval` (54 questions, 127 chunks, Apple Silicon):

|                 | hit@1 | hit@3 | Time for each question |
| --------------- | ----- | ----- | ---------------------- |
| Vector only     | 48/54 | 52/54 | —                      |
| Vector + rerank | 48/54 | 54/54 | +2.1 s                 |

Rerank finds the correct chunk in the top 3 for all questions. But it also moves some correct chunks down from position 1, so hit@1 does not change.
A test set of 54 questions is small. A difference of 1 or 2 questions can be chance.
With only 2 documents (11 chunks), rerank was not useful: 20 candidates were all the chunks, and all chunks fit in the prompt.

## Not included on purpose (possible later additions)

- Hybrid search (BM25 keywords and vectors): vector search often fails on proper nouns and ID numbers.
- PDF and Word parsing, multi-turn conversation, a web UI.
- A local LLM for generation (for example, `ollama pull qwen3:8b`): generation also runs offline, and no data leaves the company.
