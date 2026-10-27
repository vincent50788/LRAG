"""Step 1, chunking: split the documents into chunks for search.

A chunk is one part of a document. The system stores and finds each chunk as one unit.
If a chunk is too large, unrelated text dilutes its vector. If a chunk is too small, it loses its context.
"""
import re


def split(text, source):
    """Split the text at each `##` heading. Add "document > section" to the start of each chunk.
    This prefix gives the vector the context of the chunk."""
    title = text.splitlines()[0].lstrip("# ").strip()
    chunks = []
    for block in re.split(r"\n(?=## )", text)[1:]:  # Item [0] is the document title and the introduction.
        heading, _, body = block.partition("\n")
        chunks.append({"source": source, "text": f"{title} > {heading.lstrip('# ').strip()}\n{body.strip()}"})
    # ponytail: Each section becomes one chunk. If sections are long, use fixed-length chunks with overlap.
    return chunks


def load_chunks(data_dir):
    """Read all .md files in the folder and split them into chunks: [{"source", "text"}, ...]."""
    return [c for f in sorted(data_dir.glob("*.md")) for c in split(f.read_text(), f.name)]
