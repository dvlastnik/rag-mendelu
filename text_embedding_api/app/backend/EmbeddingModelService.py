from sentence_transformers import SentenceTransformer
from BaseEmbeddingModelService import BaseEmbeddingModelService

class EmbeddingModelService(BaseEmbeddingModelService):
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.model = SentenceTransformer('all-MiniLM-L6-v2')
            cls._instance.installed_models = ['all-MiniLM-L6-v2']
        return cls._instance

    def __init__(self):
        pass

    def set_model(self, model_name: str):
        if not model_name:
            raise ValueError('model_name is required and cannot be blank')
        
        self.model = SentenceTransformer(model_name)
        if model_name not in self.installed_models:
            self.installed_models.append(model_name)

    def encode(self, texts):
        return self.model.encode(texts, normalize_embeddings=True)
    
    def get_installed_models(self) -> list[str]:
        return self.installed_models
    
    def get_current_model(self):
        return self.model._model_card
