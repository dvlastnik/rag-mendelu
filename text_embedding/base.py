from abc import ABC, abstractmethod
from typing import List


class BaseDenseEmbeddingLibrary(ABC):
    @abstractmethod
    def encode(self, texts: List[str]) -> List[List[float]]:
        pass

    @abstractmethod
    def set_model(self, model_name: str):
        pass

    @abstractmethod
    def get_current_model(self) -> str:
        pass

    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Return the output dimensionality of the current dense model."""
        pass
