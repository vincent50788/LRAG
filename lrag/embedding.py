"""Step 2, embedding: change text into vectors. Text with a similar meaning gets a vector with a similar direction.

Use the same model for the questions and the chunks. If you change the model, run index again.
"""
from functools import cached_property

from chromadb import EmbeddingFunction
from sentence_transformers import SentenceTransformer

from .config import EMBED_MODEL


class Qwen3EF(EmbeddingFunction):
    """The embedding function (EF) that Chroma uses.
    Qwen3-Embedding is asymmetric. It encodes a chunk as it is. It adds the `query` prefix to a question.
    The Chroma built-in SentenceTransformerEmbeddingFunction does not override embed_query.
    With that class, questions do not get the prefix."""

    def __init__(self, model_name=EMBED_MODEL):
        self.model_name = model_name  # Do not use self.name. EmbeddingFunction already has a name() method.

    @cached_property
    def model(self):  # Loads the model (~1.1 GB) only at its first use.
        try:  # Read the local cache first, with no network access. This saves the update check (~7 s) on each run.
            return SentenceTransformer(self.model_name, local_files_only=True)
        except OSError:  # No local copy. Download the model from Hugging Face to ~/.cache/huggingface/.
            return SentenceTransformer(self.model_name)

    def __call__(self, input):  # Chroma calls this in add() for chunks. No prefix.
        return self.model.encode(list(input)).tolist()

    def embed_query(self, input):  # Chroma calls this in query(query_texts=...) for questions. Adds the prefix.
        return self.model.encode(list(input), prompt_name="query").tolist()
