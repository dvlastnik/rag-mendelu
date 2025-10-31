from typing import List
from sentence_transformers import util

from embedding.EmbeddingModelService import EmbeddingModelService

class SentenceSimilarity:
    def __init__(self, embedding_service: EmbeddingModelService, similarity_threshold=0.3):
        self.embedding_service = embedding_service
        self.similarity_threshold = similarity_threshold

    def similarities(self, sentences: List[str]):
        embeddings_generator = self.embedding_service.encode(text=sentences)
        embeddings = [embedding for embedding in embeddings_generator]

        similarities = []
        for i in range(1, len(embeddings)):
            similarity = util.pytorch_cos_sim(embeddings[i-1], embeddings[i]).item()
            similarities.append(similarity)

        return similarities