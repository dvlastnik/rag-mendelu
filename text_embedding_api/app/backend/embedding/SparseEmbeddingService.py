from fastembed import SparseTextEmbedding
from typing import List, Dict, Any


class SparseEmbeddingService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")
            cls._instance.installed_models = ['all-MiniLM-L6-v2']
        return cls._instance

    def __init__(self):
        pass

    def embed(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Returns a list of dictionaries suitable for API response.
        Each dict contains 'indices' and 'values'.
        """
        results = list(self.model.embed(texts))
        
        serialized = []
        for res in results:
            serialized.append({
                "indices": res.indices.tolist(),
                "values": res.values.tolist()
            })
        return serialized