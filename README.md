# LRAG: a local RAG system for an employee handbook (for learning)

- **Search** runs on this computer: [sentence-transformers](https://sbert.net) with `Qwen/Qwen3-Embedding-0.6B`, then `Qwen/Qwen3-Reranker-0.6B` sorts the results again. The handbook does not leave the computer.
- **Generation** runs in the cloud: the Gemini API free tier. The system sends only the question and the top 3 chunks. (Google can use free-tier content to improve its products. For real company documents, use a paid tier or a local model.)

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
export GEMINI_API_KEY=...                # Get a free key at aistudio.google.com.
.venv/bin/python rag.py index            # The first run downloads the embedding model (about 1.1 GB).
.venv/bin/python rag.py ask "What do I do if I forget to clock in?"   # The first run downloads the reranker (about 1.2 GB).
.venv/bin/python rag.py eval             # Measure retrieval with and without rerank.
brew install tesseract && .venv/bin/python rag.py ocr   # Optional: read the sample scan in scans/ with OCR. It writes data/*.md. Then run index again.
```

The 14 handbook documents in `data/` (127 chunks) are for a fictional company, Starriver Tech. Three more files in `data/` (6 chunks) are OCR output from the files in `scans/`: see "OCR input" below. The models go to the Hugging Face cache, `~/.cache/huggingface/hub/`, not to this folder.

## Requirements

- Python 3 with `venv`. The code was run on Python 3.14, on Apple Silicon.
- About 2.5 GB of disk space for the two local models (1.1 GB for the embedding model, 1.2 GB for the reranker). Both download on first use.
- A Gemini API key, only for `ask`. `index`, `search`, `eval`, and `ocr` do not call the cloud.
- Tesseract, only for `ocr`: `brew install tesseract` on macOS. The Python packages for OCR (`pytesseract`, `pypdfium2`, `pillow`) are in `requirements.txt`, but the code loads them only when `ocr` runs.

## Commands

| Command                       | What it does                                                                                                                  | Needs the key? |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------- | -------------- |
| `rag.py index`                | Reads `data/*.md`, splits, embeds, and stores the chunks in `chroma_db/`. It deletes the old index and builds a new one.      | No             |
| `rag.py search "question"`    | Shows the top 3 chunks and their scores. Does not call the LLM.                                                               | No             |
| `rag.py ask "question"`       | Full RAG: search, rerank, build the prompt, and stream the answer from Gemini.                                                | Yes            |
| `rag.py eval [file.json]`     | Measures hit@1 and hit@3, with and without rerank. The default file is `eval/questions.json`.                                 | No             |
| `rag.py ocr [file-or-folder]` | Reads scanned PDFs and images (default folder: `scans/`) with Tesseract and writes `data/<name>.md`. Run `index` after it.    | No             |

Any other input prints the help text.

## Configuration

All settings are in `lrag/config.py`. These can be changed with environment variables:

| Variable         | Default                     | Meaning                                                                                 |
| ---------------- | --------------------------- | --------------------------------------------------------------------------------------- |
| `GEMINI_API_KEY` | (none)                      | The key for the Gemini API. Only `ask` needs it.                                        |
| `EMBED_MODEL`    | `Qwen/Qwen3-Embedding-0.6B` | The embedding model. If you change it, run `index` again.                               |
| `RERANK_MODEL`   | `Qwen/Qwen3-Reranker-0.6B`  | The reranker model.                                                                     |
| `RERANK`         | `1`                         | `RERANK=0` turns off rerank. Search then uses only the vector order.                    |
| `CHAT_MODEL`     | `gemini-flash-latest`       | The Gemini model for the answer. If it fails, the code tries `gemini-flash-lite-latest`. |
| `OCR_LANG`       | `eng`                       | The Tesseract language. Join several with `+`, for example `eng+chi_tra`.               |
| `OCR_DPI`        | `300`                       | The resolution to draw a scanned PDF page before OCR.                                   |

`TOP_K` (3 chunks in the prompt) and `CANDIDATES` (20 chunks from the vector search) are constants in `lrag/config.py`, not environment variables.

## Privacy: what leaves the computer

| Step                                                  | Where it runs      | What is sent out                                  |
| ----------------------------------------------------- | ------------------ | ------------------------------------------------- |
| OCR, chunking, embedding, vector search, rerank, eval | This computer      | Nothing                                           |
| `ask`: generation                                     | Gemini API (cloud) | The question and the top 3 chunks, in one prompt. |

The full documents and the index never leave the computer. The first run of each model downloads it from Hugging Face. After that, everything except `ask` works offline.

## What RAG does

An LLM did not see your company handbook during training. RAG (Retrieval-Augmented Generation) **finds the related text first, then puts that text in the prompt**. The LLM then answers from that text.

A **chunk** is one part of a document. The system stores, finds, and sends each chunk as one unit.

```
Offline (ocr)    scans/*.pdf, *.png ─OCR─▶ data/*.md   (optional: only for scans and images)
Offline (index)  data/*.md ─split─▶ chunks ─embed─▶ vectors ─▶ chroma_db/
Online (ask)     question ─embed─▶ vector ─cosine similarity─▶ top 20 chunks ─rerank─▶ top 3 chunks ─add to prompt─▶ LLM ─▶ answer
```

| Step         | Location                          | Key points                                                                                                                                                                                                                        |
| ------------ | --------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Chunking     | `lrag/chunking.py` `split()`      | Splits the text at each `##` heading. If a chunk is too large, unrelated text dilutes its vector. If a chunk is too small, it loses its context.                                                                                  |
| Embedding    | `lrag/embedding.py` `Qwen3EF`     | Sentences with a similar meaning get vectors with a similar direction. Use the **same** model for questions and chunks. If you change the model, run `index` again. Qwen3 needs the `query` prompt for questions, not for chunks. |
| Retrieval    | `lrag/vector_store.py` `search()` | Sorts the chunks by cosine similarity and takes the top 20 candidates. See the Chroma section below.                                                                                                                              |
| Rerank       | `lrag/rerank.py` `rerank()`       | A cross-encoder reads the question and each candidate together, and keeps the best 3\. More accurate than vectors, but slower. See the Rerank section below. `RERANK=0` turns it off.                                             |
| Augmentation | `lrag/llm.py` `SYSTEM_PROMPT`     | Gives the model three rules: use only the reference material, give the source, and say "I do not know" when the answer is missing. These rules reduce hallucination.                                                              |
| Generation   | `lrag/llm.py` `generate()`        | Calls the Gemini API and prints the answer as a stream. Retrieval and generation can use models from different vendors.                                                                                                           |

## Project structure

```
rag.py              Command-line entry point: index / search / ask / eval / ocr
lrag/
  config.py         Settings: paths, model names, Top-K, rerank (environment variables can override the models)
  timing.py         tick(): prints the time of each step
  ocr.py            Step 0: OCR. Reads scans/ with Tesseract and writes data/*.md (not a RAG step)
  chunking.py       Step 1: chunking
  embedding.py      Step 2: embedding (Qwen3EF)
  vector_store.py   Step 3: Chroma index and search
  rerank.py         Step 3b: rerank the candidates with Qwen3-Reranker
  llm.py            Step 4: prompt and Gemini generation
  pipeline.py       Connects steps 1–4 into build_index() / search() / ask()
  evaluate.py       `rag.py eval`: hit@1 and hit@3, with and without rerank
data/               The documents to search (*.md): 14 handbook files, and the OCR output of scans/
scans/              Inputs for OCR: the sample PDF and its script (make_sample_scan.py), and two sample images
eval/
  questions.json    54 test questions about the handbook, with the correct chunk for each one
  ocr_questions.json  11 test questions about the OCR files (10 about the travel guide, 1 about the image)
chroma_db/          The vector index (`rag.py index` makes it, git ignores it)
requirements.txt    google-genai, sentence-transformers, chromadb, and the optional OCR packages
```

### Data flow in code

`rag.py index` → `pipeline.build_index()` → `chunking.load_chunks()` → `vector_store.build()` (the `Qwen3EF` embedding function makes the vectors).
`rag.py ask` → `pipeline.ask()` → `vector_store.search()` (20 candidates) → `rerank.rerank()` (best 3) → `llm.generate()`.
`rag.py eval` calls `vector_store.search()` and `rerank.rerank()` directly, and it does not call the LLM.

### Test set format

`eval/*.json` is a list of questions. `expected` is the first line of the correct chunk: "document title > section heading".

```json
[{"q": "When does my salary arrive each month?", "expected": "Starriver Tech Compensation and Payroll Policy > Pay Day"}]
```

A question counts as a hit@k when its `expected` chunk is in the first k results. If you rename a heading in `data/`, update the `expected` text too.

## Experiments

1. Run `RERANK=0 .venv/bin/python rag.py search "Can I use my own phone for work?"`. "Device Use" is not the first result. Other chunks also talk about phones, accounts, and work outside the office. The Device Use chunk contains the answer (BYOD), but it also contains other topics, such as laptop encryption and software. The correct chunk is in position 3\. Top-3 retrieval still finds it. **This is why K is not 1.** Then run the same search without `RERANK=0`. The reranker moves Device Use to position 1.
2. Change `split()` in `lrag/chunking.py` to split at each line. Run `index` again. Then search the same question again and compare the scores.
3. Run `ask "Does the company give parking spaces to employees?"`. The handbook does not answer this. Check if the model says that it does not know.
4. Remove the rules from `SYSTEM_PROMPT` in `lrag/llm.py`. Ask question 3 again and look for hallucination.
5. Add a new .md file to `data/` and run `index` again. You can then ask about the new content with no code change.
6. Compare embedding models: run `EMBED_MODEL=Qwen/Qwen3-Embedding-4B .venv/bin/python rag.py index`, then run `search`. **Do not compare scores from different models directly.** Compare the gap between result 1 and result 2\. A larger gap shows that the model separates the correct chunk better.
7. Search with a question in a different language, for example Spanish. A multilingual embedding model can search across languages.
8. Run `rag.py eval`. Look at the questions that go down with rerank, not only the ones that go up. Then add your own questions to `eval/questions.json`. Without a test set, you cannot know if a change helps.

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

| Method          | hit@1 | hit@3 | Time for each question |
| --------------- | ----- | ----- | ---------------------- |
| Vector only     | 48/54 | 52/54 | --                     |
| Vector + rerank | 48/54 | 54/54 | +2.1 s                 |

Rerank finds the correct chunk in the top 3 for all questions. But it also moves some correct chunks down from position 1, so hit@1 does not change. A test set of 54 questions is small. A difference of 1 or 2 questions can be chance. With only 2 documents (11 chunks), rerank was not useful: 20 candidates were all the chunks, and all chunks fit in the prompt.

## OCR input (`lrag/ocr.py`)

OCR (Optical Character Recognition) reads the text in a picture. It is not a RAG step. It runs **before** chunking, it only makes text, and it changes nothing else in the pipeline.

```
scans/*.pdf, *.png, *.webp ─Tesseract─▶ data/<name>.md ─▶ index ─▶ ask   (index and ask work as before)
```

- Tesseract is a program, not a Python package. Install it with `brew install tesseract`. It reads English by default. For another language, install its data (`brew install tesseract-lang`) and set `OCR_LANG`, for example `OCR_LANG=eng+chi_tra`.
- `rag.py ocr [file-or-folder]` reads one file or a folder (default: `scans/`). A PDF page that has a text layer is copied, because OCR would only add errors. A page with no text layer is drawn as a picture and read by OCR. An image (PNG, JPG, WebP, TIFF) always goes to OCR.
- OCR text has no headings, but the chunker needs `##` headings. So the loader writes one `## Page N` for each page, and each page becomes one chunk. The page number also gives each answer a source: "Overseas Travel Guide > Page 2".
- The output is a normal file in `data/`. Open it and look for OCR mistakes (the sample has one: "lf" instead of "If"). You can fix them by hand. The comment at the top records the source file and the settings. OCR is a separate command on purpose: you can check the text before it goes into the index, and `index` does not run OCR again each time.
- A page is a coarse chunk. If your pages are dense, try to split each page into paragraphs and compare the `eval` results.
- The sample is `scans/starriver-tech-overseas-travel-guide.pdf`: 4 pages, no text layer, a small tilt, blur, and noise. `scans/make_sample_scan.py` makes it, and the text in that script is the right answer for OCR. The guide gives the overseas limits that the handbook only mentions ("Different limits apply to overseas regions").
- Run the 10 questions about the guide with `rag.py eval eval/ocr_questions.json`. The correct chunks in that file use the headings that the loader writes (`... > Page 2`), not text that OCR read. If the key were OCR text, an OCR mistake in it would look like a retrieval miss.

### Two sample images

`scans/` also has two images that are not part of the Starriver story. They test the image path (an image always goes to OCR, and always makes one `## Page 1`):

- `she_sells_sea_shell.jpg`: a children's poster with a cartoon title and large handwriting-style text. OCR reads the poem well, but it misreads the decorative title ("Ssastotin" instead of "Seashells").
- `safety_check_word.jpg`: a logo with only the word "SIMPLE". It makes a chunk with one word. `eval/ocr_questions.json` has one question for it ("what's the word for saftey check?", the typo is on purpose).

Both are only samples. They are not about the company, and a real handbook would not have them. Delete their `.md` files from `data/` and run `index` again if you do not want them in the index.

### Results on the travel guide

These were measured before the two images were added (10 questions, 131 chunks, PDF read at 300 DPI). `eval/ocr_questions.json` now has 11 questions and the index has 133 chunks, so run `rag.py eval eval/ocr_questions.json` to get the current numbers.

| Method          | hit@1 | hit@3 |
| --------------- | ----- | ----- |
| Vector only     | 8/10  | 10/10 |
| Vector + rerank | 10/10 | 10/10 |

The 54 handbook questions give the same results as before, with the 4 new chunks in the index.

### Experiment: make the OCR worse

`OCR_DPI` is the resolution that `rag.py ocr` uses to draw a scanned PDF page (default 300). Lower it, run `rag.py ocr`, then `index`, then `eval eval/ocr_questions.json`. At the end, run `rag.py ocr` again without `OCR_DPI` to restore the text.

| OCR_DPI | Text compared with the 300 DPI text | Vector only (hit@1, hit@3) | Vector + rerank (hit@1, hit@3) |
| ------- | ----------------------------------- | -------------------------- | ------------------------------ |
| 300     | the reference                       | 8/10, 10/10                | 10/10, 10/10                   |
| 100     | 99.8% the same                      | 8/10, 10/10                | 10/10, 10/10                   |
| 72      | 97.7% the same                      | 9/10, 10/10                | 10/10, 10/10                   |
| 50      | 47.1% the same                      | 7/10, 10/10                | 8/10, 10/10                    |
| 36      | 10.7% the same                      | 4/10, 9/10                 | 2/10, 6/10                     |

("The same" is the `difflib` ratio between the two OCR texts. It is not an accuracy against the original text.)

- At 72 DPI, the text has visible mistakes ("Startiver", "approvesit"), but retrieval does not get worse. The embedding encodes the meaning of the whole chunk, so a few wrong letters change the vector very little. hit@1 even goes from 8/10 to 9/10. With 10 questions, treat that as chance.
- At 50 DPI, about half of the text is different, and the correct page is still in the top 3 for all 10 questions.
- Retrieval breaks only at 36 DPI, when almost all of the text is lost. Then the reranker is **worse** than the vector search (6/10 against 9/10 at hit@3). A reranker that is better on clean text can be worse on garbage. With 10 questions, this is a hint, not a proof.
- hit@k checks only that the right page is found. It does not check that the numbers on it are right. If OCR changes "NT$4,500" to "NT$4,800", retrieval still works, but the answer is wrong. When numbers matter, compare the answer with the scan.

## Troubleshooting

| Problem                                              | Cause and fix                                                                                                                         |
| ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `ask` fails with an API key error                    | Set `GEMINI_API_KEY` in the same shell. `search` and `eval` do not need it.                                                           |
| `ask` is slow, or the model is busy (HTTP 429 / 503) | The free tier is often busy. The code retries with `gemini-flash-lite-latest`. Wait a minute and try again.                           |
| Search results look wrong after a model change       | The index has vectors from the old model. Run `index` again.                                                                          |
| New or changed file in `data/` has no effect         | `index` is a separate step. Run it again. `ocr` does not update the index either.                                                     |
| `Tesseract is not installed`                         | Run `brew install tesseract`. It is a program, not a Python package.                                                                  |
| `Missing package (pytesseract...)`                   | Run `.venv/bin/pip install -r requirements.txt`.                                                                                      |
| OCR text has strange letters or words                | Normal for OCR. Open `data/<name>.md` and fix it by hand, or read the scan with a higher `OCR_DPI`. Then run `index` again.           |
| `eval` stops with a JSON error                       | Check the commas in your questions file. Every item needs a comma after it, except the last.                                          |
| The first command is very slow                       | It downloads the model (about 1 GB). Later runs use the cache in `~/.cache/huggingface/hub/`.                                         |

## Not included on purpose (possible later additions)

- Hybrid search (BM25 keywords and vectors): vector search often fails on proper nouns and ID numbers.
- Word and other file types (the OCR step reads only PDFs and images), multi-turn conversation, a web UI.
- Layout-aware OCR: Tesseract reads tables and multiple columns in the wrong order. If your scans have them, a tool such as [Docling](https://docling.ai) is the next step.
- A local LLM for generation (for example, `ollama pull qwen3:8b`): generation also runs offline, and no data leaves the company.
