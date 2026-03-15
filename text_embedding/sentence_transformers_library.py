from typing import List

from text_embedding.base import BaseDenseEmbeddingLibrary

DEFAULT_MODEL = "all-MiniLM-L6-v2"


class SentenceTransformersLibrary(BaseDenseEmbeddingLibrary):
    def __init__(self, model_name: str = DEFAULT_MODEL):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Run `uv add sentence-transformers` to use this backend."
            )
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def set_model(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def get_current_model(self) -> str:
        return self.model_name

    def get_embedding_dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()
