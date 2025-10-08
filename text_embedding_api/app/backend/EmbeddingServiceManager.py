from EmbeddingModelService import EmbeddingModelService
from FastEmbeddingModelService import FastEmbedEmbeddingService

class EmbeddingServiceManager:
    def __init__(self):
        self.services = {
            "sentence_transformers": EmbeddingModelService(),
            "fastembed": FastEmbedEmbeddingService()
        }
        self.active_service_name = "fastembed"
        self.active_service = self.services[self.active_service_name]

    def set_active_service(self, service_name: str):
        if service_name not in self.services.keys():
            raise ValueError(f"Unknown embedding service: {service_name}")
        self.active_service_name = service_name
        self.active_service = self.services[service_name]

    def encode(self, text: str):
        return self.active_service.encode(text)
    
    def get_installed_models(self):
        models = {}
        for name, service in self.services.items():
            models[name] = service.get_installed_models()
        return models