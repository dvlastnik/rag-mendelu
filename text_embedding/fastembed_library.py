from typing import List

from fastembed import TextEmbedding

from text_embedding.base import BaseDenseEmbeddingLibrary

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


class FastEmbedLibrary(BaseDenseEmbeddingLibrary):
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.model = TextEmbedding(model_name=model_name)

    def encode(self, texts: List[str]) -> List[List[float]]:
        return [embedding.tolist() for embedding in self.model.embed(texts)]

    def set_model(self, model_name: str):
        self.model_name = model_name
        self.model = TextEmbedding(model_name=model_name)

    def get_current_model(self) -> str:
        return self.model_name
