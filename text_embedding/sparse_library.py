from typing import List

from fastembed import SparseTextEmbedding

from text_embedding.models import SparseVectorData

DEFAULT_MODEL = "prithivida/Splade_PP_en_v1"


class SparseEmbeddingLibrary:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.model = SparseTextEmbedding(model_name=model_name)

    def embed(self, texts: List[str]) -> List[SparseVectorData]:
        results = list(self.model.embed(texts))
        return [
            SparseVectorData(indices=r.indices.tolist(), values=r.values.tolist())
            for r in results
        ]
