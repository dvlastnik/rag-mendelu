import traceback
from typing import List, Dict, Any, Type
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams, PointStruct

from database.base.MyDocument import MyDocument
from database.base.DbOperationResult import DbOperationResult
from database.base.BaseDbRepository import BaseDbRepository

class QdrantDbRepository(BaseDbRepository):
    name = 'qdrant'

    def __init__(self, ip: str, port: int, grpc_port: int = 6334, collection_name: str = "", metadata: Dict = {}):
        """
        Initializes the Qdrant repository.
        
        Args:
            ip: Host IP for Qdrant.
            port: gRPC port for Qdrant (e.g., 6333).
            collection_name: Name of the collection.
            metadata: Must contain {'vector_size': int, 'distance': str}
                      e.g., {'vector_size': 768, 'distance': 'COSINE'}
        """
        super().__init__(ip, port, collection_name, metadata)
        self.client: QdrantClient | None = None
        self.grpc_port = grpc_port
        
        self.vector_size = self.metadata.get('vector_size')
        if self.vector_size is None or not isinstance(self.vector_size, int):
            raise ValueError("QdrantDbRepository 'metadata' must include 'vector_size' (e.g., {'vector_size': 768})")
            
        distance_str = self.metadata.get('distance', 'DOT').upper()
        self.distance = getattr(models.Distance, distance_str, models.Distance.DOT)
        self.logger.info(f"QdrantRepository configured for collection '{collection_name}' with size {self.vector_size} and distance {self.distance}")

    def connect(self) -> DbOperationResult:
        """Connects to the Qdrant gRPC client."""
        try:
            self.logger.info(f"Connecting to Qdrant at {self.ip}:{self.port}...")
            # Using gRPC port by default as it's faster for service-to-service
            self.client = QdrantClient(host=self.ip, port=self.port, grpc_port=self.grpc_port)
            self.logger.info("Qdrant connection successful.")
            return DbOperationResult(success=True)
        except Exception as e:
            self.logger.error(f"Failed to connect to Qdrant: {e}")
            return DbOperationResult(success=False, message=str(e))

    def if_collection_exist_delete(self) -> DbOperationResult:
        """Checks if the collection exists and deletes it if it does."""
        if not self.client:
            return DbOperationResult(success=False, message="Client not connected")
        try:
            if self.client.collection_exists(self.collection_name):
                self.logger.warning(f"Collection '{self.collection_name}' already exists. Deleting.")
                self.client.delete_collection(self.collection_name)
            return DbOperationResult(success=True)
        except Exception as e:
            self.logger.error(f"Failed to delete collection: {e}")
            return DbOperationResult(success=False, message=str(e))

    def create_collection(self) -> DbOperationResult:
        """Creates a new collection with the specified vector config."""
        if not self.client:
            return DbOperationResult(success=False, message="Client not connected")
        try:
            self.logger.info(f"Creating collection '{self.collection_name}'...")

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": models.VectorParams(
                        size=self.vector_size,
                        distance=self.distance
                    )
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(
                        index=models.SparseIndexParams(
                            on_disk=True
                        )
                    )
                }
            )
            self.logger.info("Collection created successfully.")
            return DbOperationResult(success=True)
        except Exception as e:
            self.logger.error(f"Failed to create collection: {e}")
            return DbOperationResult(success=False, message=str(e))

    def search(
        self, 
        text: str | List[str] | None = None,
        text_embedded: List[float] | None = None,
        sparse_embedded: Any | None = None,
        n_results: int = 3
    ) -> DbOperationResult:
        """
        Performs Hybrid Search.
        Note: You need to pass the sparse query vector here too now.
        """
        if not self.client:
            return DbOperationResult(success=False, message="Client not connected")
        
        if text_embedded is None:
            return DbOperationResult(success=False, message="Search requires 'text_embedded' (dense vector).")

        try:
            search_query = text_embedded
            prefetch = None

            if sparse_embedded:
                prefetch = [
                    models.Prefetch(
                        query=models.SparseVector(
                            indices=sparse_embedded.indices,
                            values=sparse_embedded.values
                        ),
                        using="sparse",
                        limit=n_results * 2
                    ),
                    models.Prefetch(
                        query=text_embedded,
                        using="dense",
                        limit=n_results * 2
                    ),
                ]
                search_query = models.FusionQuery(fusion=models.Fusion.RRF)

            search_result = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=prefetch,
                query=search_query if prefetch else text_embedded,
                using="dense" if not prefetch else None,
                limit=n_results,
                with_payload=True
            )
            
            documents = []
            for point in search_result.points:
                metadata = point.payload.copy()
                metadata['score'] = point.score
                documents.append(
                    MyDocument(
                        id=point.id,
                        text=point.payload.get('text', ''), 
                        embedding=[], 
                        metadata=metadata
                    )
                )
            
            return DbOperationResult(success=True, data=documents)
        except Exception as e:
            self.logger.error(f"Failed to search: {e}")
            traceback.print_exc()
            return DbOperationResult(success=False, message=str(e))

    def get_count(self) -> int:
        """Gets the exact count of vectors in the collection."""
        if not self.client:
            self.logger.error("Client not connected, returning count 0")
            return 0
        try:
            count_result = self.client.count(self.collection_name, exact=True)
            return count_result.count
        except Exception as e:
            self.logger.error(f"Failed to get count: {e}")
            return 0

    def check_if_data_were_inserted(self) -> DbOperationResult:
        """Checks if the count is greater than 0."""
        count = self.get_count()
        if count > 0:
            return DbOperationResult(success=True, data=count)
        else:
            return DbOperationResult(success=False, message="No data found.", data=0)

    def insert(self, docs: List[MyDocument]) -> DbOperationResult:
        """Inserts a list of MyDocument objects into the collection."""
        if not self.client:
            return DbOperationResult(success=False, message="Client not connected")
        
        points = []
        for doc in docs:
            if not doc.embedding:
                self.logger.warning(f"Skipping doc {doc.id}, no embedding found.")
                continue

            vector_payload = {
                "dense": doc.embedding
            }

            if doc.sparse_embedding:
                vector_payload["sparse"] = models.SparseVector(
                    indices=doc.sparse_embedding.indices,
                    values=doc.sparse_embedding.values
                )
                
            point = models.PointStruct(
                id=str(doc.id),
                vector=vector_payload,
                payload=doc.metadata
            )
            points.append(point)
            
        if not points:
            return DbOperationResult(success=False, message="No valid documents with embeddings to insert.")
            
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True
            )
            return DbOperationResult(success=True)
        except Exception as e:
            self.logger.error(f"Failed to insert points: {e}")
            traceback.print_exc()
            return DbOperationResult(success=False, message=str(e))

    def delete(self, ids: List[str]) -> DbOperationResult:
        """Deletes points from the collection by their IDs."""
        if not self.client:
            return DbOperationResult(success=False, message="Client not connected")
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=ids),
                wait=True
            )
            return DbOperationResult(success=True)
        except Exception as e:
            self.logger.error(f"Failed to delete points: {e}")
            return DbOperationResult(success=False, message=str(e))

    def close(self) -> DbOperationResult:
        """Closes the client connection."""
        try:
            if self.client:
                self.client.close()
                self.logger.info("Qdrant connection closed.")
            return DbOperationResult(success=True)
        except Exception as e:
            return DbOperationResult(success=False, message=str(e))