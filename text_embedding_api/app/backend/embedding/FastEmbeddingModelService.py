from typing import List
from fastembed.embedding import TextEmbedding
from embedding.BaseEmbeddingModelService import BaseEmbeddingModelService

class FastEmbedEmbeddingService(BaseEmbeddingModelService):
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.installed_models = [
                "BAAI/bge-small-en-v1.5",
                "nomic-ai/nomic-embed-text-v1",
                "intfloat/e5-small-v2"
            ]
            cls._instance.model_name = cls._instance.installed_models[0]
            cls._instance.model = TextEmbedding(model_name=cls._instance.model_name)
        return cls._instance

    def __init__(self):
        pass

    def set_model(self, model_name: str):
        if not model_name or model_name not in self.installed_models:
            raise ValueError(f"Invalid model_name: {model_name}")
        self.model_name = model_name
        self.model = TextEmbedding(model_name=model_name)

    def encode(self, text: List[str] | str):
        if isinstance(text, list):
            return self.model.embed(text)
        return list(self.model.embed([text]))[0]

    def get_installed_models(self):
        return self.installed_models
    
    def get_current_model(self):
        return self.model.model_name
