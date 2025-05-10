import chromadb

from TextEmbeddingService import TextEmbeddingService

class ChromaDbModel:
    def __init__(self, id: str, metadata: dict, text: str, text_embedding: list[int]):
        self.id = id
        self.metadata = metadata
        self.text = text
        self.text_embedding = text_embedding

    @staticmethod
    def from_document(document, embedding_service: TextEmbeddingService) -> 'ChromaDbModel':
        response = embedding_service.get_embedding_with_uuid(data=document.page_content)
        response = response[0]

        return ChromaDbModel(id=response.uuid, metadata=document.metadata, text=document.page_content, text_embedding=response.embedding)