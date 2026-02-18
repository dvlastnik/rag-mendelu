import numpy as np
from typing import List
from text_embedding_api.TextEmbeddingService import TextEmbeddingService

class SentenceSimilarity:
    def __init__(self, embedding_service: TextEmbeddingService, similarity_threshold=0.3):
        self.embedding_service = embedding_service
        self.similarity_threshold = similarity_threshold

    def similarities(self, sentences: List[str]):
        embeddings_response = self.embedding_service.get_embedding_with_uuid(sentences, 50)
        embeddings = [np.array(embedding.embedding) for embedding in embeddings_response]

        similarities = []
        for i in range(1, len(embeddings)):
            vec1 = embeddings[i-1]
            vec2 = embeddings[i]
            
            dot_product = np.dot(vec1, vec2)
            norm_a = np.linalg.norm(vec1)
            norm_b = np.linalg.norm(vec2)
            
            if norm_a == 0 or norm_b == 0:
                similarity = 0.0
            else:
                similarity = dot_product / (norm_a * norm_b)
                
            similarities.append(float(similarity))

        return similarities