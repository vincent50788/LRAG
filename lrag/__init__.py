"""LRAG: a local RAG system for the employee handbook. Each file does one RAG step:

    config.py        Settings: paths, model names, Top-K.
    timing.py        Timing: tick() prints the time of each step.
    chunking.py      Step 1, chunking: reads data/*.md and splits the text at each `##` heading.
    embedding.py     Step 2, embedding: Qwen3-Embedding encodes chunks and questions in different ways.
    vector_store.py  Step 3, storage: the Chroma vector store. build() makes the index. search() finds chunks.
    rerank.py        Step 3b, rerank: a cross-encoder sorts the vector search candidates again and keeps the best.
    llm.py           Step 4, generation: builds the prompt and streams the answer from Gemini.
    pipeline.py      Connects the steps: build_index(), search(), ask().
    evaluate.py      Measures retrieval (hit@k) with eval/questions.json, with and without rerank.
"""
