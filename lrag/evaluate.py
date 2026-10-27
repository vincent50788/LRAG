"""Measure retrieval: does the correct chunk come back? Compare the vector order with the rerank order.

eval/questions.json has questions and the correct chunk for each one ("document > section").
For each question, one vector search gets CANDIDATES chunks:
- Vector only: the order of the vector search.
- Vector + rerank: the reranker sorts the same candidates again.
hit@k = the number of questions with the correct chunk in the top k. Only the top TOP_K go to the LLM.
"""
import json
import time

from . import rerank, vector_store
from .config import CANDIDATES, EVAL_FILE, TOP_K


def _rank(expected, hits):
    """The position (1, 2, ...) of the correct chunk in hits, or None if it is not there."""
    heads = [c["text"].splitlines()[0] for _, c in hits]
    return heads.index(expected) + 1 if expected in heads else None


def run():
    questions = json.loads(EVAL_FILE.read_text())
    rows, rerank_time = [], 0.0
    for item in questions:
        hits = vector_store.search(item["q"], CANDIDATES)
        start = time.perf_counter()
        reranked = rerank.rerank(item["q"], hits, CANDIDATES)  # Keep all candidates to see the full new order.
        rerank_time += time.perf_counter() - start
        rows.append((item["q"], _rank(item["expected"], hits), _rank(item["expected"], reranked)))

    n = len(rows)
    hit = lambda col, k: sum(1 for r in rows if r[col] and r[col] <= k)  # noqa: E731
    print(f"\n{n} questions. Rerank sorts the top {CANDIDATES} of the vector search. Average rerank time: {rerank_time / n:.2f}s.\n")
    print(f"{'':18}hit@1   hit@{TOP_K}")
    print(f"{'Vector only':18}{hit(1, 1):>2}/{n}  {hit(1, TOP_K):>2}/{n}")
    print(f"{'Vector + rerank':18}{hit(2, 1):>2}/{n}  {hit(2, TOP_K):>2}/{n}")
    print(f"\nThe correct chunk is in the top {CANDIDATES} for {hit(1, CANDIDATES)}/{n} questions. Rerank cannot fix the others.")

    changed = [r for r in rows if r[1] != r[2]]
    if changed:
        print('\nQuestions where the rank of the correct chunk changed (vector → rerank, "-" = not found):')
        for q, v, r in sorted(changed, key=lambda r: (r[2] or 99) - (r[1] or 99)):
            mark = "↑" if (r or 99) < (v or 99) else "↓"
            print(f"  {mark} {v or '-':>2} → {r or '-':<2} {q}")
