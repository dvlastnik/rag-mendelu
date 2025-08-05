from abc import ABC, abstractmethod
from typing import List

class BaseEmbeddingModelService(ABC):
    """
    This class is only abstraction for implementations of Embedding service class.
    """
    @abstractmethod
    def set_model(self, model_name: str):
        pass

    @abstractmethod
    def encode(self, text: str):
        pass

    @abstractmethod
    def get_installed_models(self) -> List[str]:
        pass