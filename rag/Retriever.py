from langchain.schema.retriever import BaseRetriever
from langchain.schema import Document

from database.base.BaseDbRepository import BaseDbRepository
from text_embedding_api.TextEmbeddingService import TextEmbeddingService

class Retriever(BaseRetriever):
    database_service: BaseDbRepository
    embedding_service: TextEmbeddingService
    k: int = 3

    def _get_relevant_documents(self, query, *, run_manager=None):
        embedding = self.embedding_service.get_embedding_with_uuid(query)
        result = self.database_service.search(text=None, text_embedded=embedding[0].embedding, n_results=self.k)
        if not result.success:
            raise LookupError(f"Error during _get_relevant_documents: {result.message}")
        
        return [Document(page_content=doc.text, metadata=doc.metadata) for doc in result.data]