from typing import List, Optional

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

    def get_embedding_dim(self) -> int:
        info = self._get_model_info(self.model_name)
        if info and 'dim' in info:
            return info['dim']
        # Fallback: embed a test string and measure
        test = list(self.model.embed(["test"]))
        return len(test[0])

    @staticmethod
    def is_model_supported(model_name: str) -> bool:
        """Return True if *model_name* is in fastembed's known model list."""
        return any(m['model'] == model_name for m in TextEmbedding.list_supported_models())

    @staticmethod
    def _get_model_info(model_name: str) -> Optional[dict]:
        for m in TextEmbedding.list_supported_models():
            if m['model'] == model_name:
                return m
        return None
