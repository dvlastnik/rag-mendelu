from typing import List, Any, Optional
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from pydantic import PrivateAttr

# Import your internal services and types
from database.base.BaseDbRepository import BaseDbRepository
from text_embedding_api.TextEmbeddingService import TextEmbeddingService
from database.base.MyDocument import SparseVector
from utils.logging_config import get_logger

logger = get_logger(__name__)

class HybridRetriever(BaseRetriever):
    """
    A custom LangChain Retriever that performs Hybrid Search (Dense + Sparse).
    It connects the TextEmbeddingService (for query embedding) with the 
    QdrantDbRepository (for retrieval).
    """
    _database_service: BaseDbRepository = PrivateAttr()
    _embedding_service: TextEmbeddingService = PrivateAttr()
    _k: int = PrivateAttr()

    def __init__(
        self, 
        database_service: BaseDbRepository, 
        embedding_service: TextEmbeddingService, 
        k: int = 3
    ):
        """
        Args:
            database_service: The initialized Qdrant repository.
            embedding_service: The service to embed the query (Dense + SPLADE).
            k: Number of documents to retrieve.
        """
        super().__init__()
        self._database_service = database_service
        self._embedding_service = embedding_service
        self._k = k

    def _generate_hybrid_embeddings(self, query: str):
        """
        Helper to generate Dense and Sparse vectors for the query string.
        """
        response_list = self._embedding_service.get_embedding_with_uuid(data=query)

        if not response_list or len(response_list) == 0:
            raise ValueError("Embedding service returned no vectors for the query.")

        query_data = response_list[0]
        dense_vec = query_data.embedding
        sparse_vec = query_data.sparse
            
        return dense_vec, sparse_vec

    def _get_relevant_documents(
        self, 
        query: str, 
        *, 
        run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        """
        The main retrieval method called by LangChain.
        """
        try:
            dense_embedding, sparse_embedding = self._generate_hybrid_embeddings(query)
        except Exception as e:
            logger.error(f"Error generating embeddings for query: {e}")
            return []

        search_result = self._database_service.search(
            text_embedded=dense_embedding,
            sparse_embedded=sparse_embedding,
            n_results=self._k
        )
        
        if not search_result.success:
            raise LookupError(f"Error during retrieval: {search_result.message}")
        
        # 3. Convert MyDocument -> LangChain Document
        documents = []
        for doc in search_result.data:
            documents.append(
                Document(
                    page_content=doc.text, 
                    metadata=doc.metadata or {}
                ) 
            )
            
        return documents