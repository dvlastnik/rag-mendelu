import json
import traceback
from typing import List, Dict

import weaviate
from weaviate.classes.config import Configure, Property, DataType, VectorDistances, VectorFilterStrategy
from weaviate.classes.query import Filter

from database.base.MyDocument import MyDocument
from database.base.DbOperationResult import DbOperationResult
from database.base.BaseDbRepository import BaseDbRepository

class WeaviateDbRepository(BaseDbRepository):
    name = 'weaviate'

    def __init__(self, ip: str, port: int, grpc_port: int = 50051, collection_name: str = "", metadata: Dict = {}):
        """
        Initializes the Weaviate repository.
        
        Args:
            ip: Host IP for Weaviate.
            port: HTTP port for Weaviate (e.g., 8080).
            collection_name: Name of the collection (called a "Class" in Weaviate).
            metadata: Must contain {'vector_size': int, 'distance': str}
                      e.g., {'vector_size': 384, 'distance': 'COSINE'}
        """
        super().__init__(ip, port, collection_name, metadata)
        self.client: weaviate.WeaviateClient | None = None
        self.grpc_port = grpc_port
        
        # Weaviate requires vector_size and distance metric at creation time.
        self.vector_size = self.metadata.get('vector_size')
        if self.vector_size is None or not isinstance(self.vector_size, int):
            raise ValueError("WeaviateDbRepository 'metadata' must include 'vector_size' (e.g., {'vector_size': 384})")
            
        self.distance = self.metadata.get('distance', 'COSINE').upper()
        self.logger.info(f"WeaviateRepository configured for collection '{collection_name}' with size {self.vector_size} and distance {self.distance}")

    def connect(self) -> DbOperationResult:
        """Connects to the Weaviate client."""
        try:
            # Your docker-compose uses 8080 (HTTP) and 50051 (gRPC)
            # The v4 client can use both.
            self.logger.info(f"Connecting to Weaviate at http://{self.ip}:{self.port} (gRPC: {self.grpc_port})...")
            self.client = weaviate.connect_to_local(
                host=self.ip,
                port=self.port,
                grpc_port=self.grpc_port,
            )

            self.client.connect()
            if not self.client.is_ready():
                raise Exception("Weaviate client is not ready.")
            self.logger.info("Weaviate connection successful.")
            return DbOperationResult(success=True)
        except Exception as e:
            self.logger.error(f"Failed to connect to Weaviate: {e}")
            traceback.print_exc()
            return DbOperationResult(success=False, message=str(e))

    def if_collection_exist_delete(self) -> DbOperationResult:
        """Checks if the collection (class) exists and deletes it if it does."""
        if not self.client:
            return DbOperationResult(success=False, message="Client not connected")
        try:
            if self.client.collections.exists(self.collection_name):
                self.logger.warning(f"Collection '{self.collection_name}' already exists. Deleting.")
                self.client.collections.delete(self.collection_name)
            return DbOperationResult(success=True)
        except Exception as e:
            self.logger.error(f"Failed to delete collection: {e}")
            return DbOperationResult(success=False, message=str(e))

    def create_collection(self) -> DbOperationResult:
        """Creates a new collection (class) with the specified vector config."""
        if not self.client:
            return DbOperationResult(success=False, message="Client not connected")
        try:
            self.logger.info(f"Creating collection '{self.collection_name}'...")
            
            # Map distance metric names
            distance_metric = self.distance.upper()
            if distance_metric == "COSINE":
                weaviate_distance = VectorDistances.COSINE
            elif distance_metric == "DOT":
                weaviate_distance = VectorDistances.DOT
            else:
                self.logger.warning(f"Unsupported distance {distance_metric}, defaulting to COSINE.")
                weaviate_distance = VectorDistances.COSINE

            self.client.collections.create(
                name=self.collection_name,
                vector_config=Configure.Vectors.self_provided(
                    vector_index_config=Configure.VectorIndex.hnsw(
                        distance_metric=weaviate_distance,
                        filter_strategy=VectorFilterStrategy.ACORN
                    )
                ),
                properties=[
                    Property(name="text_content", data_type=DataType.TEXT),
                    Property(name="metadata_json", data_type=DataType.TEXT)
                ]
            )
            self.logger.info("Collection created successfully.")
            return DbOperationResult(success=True)
        except Exception as e:
            self.logger.error(f"Failed to create collection: {e}")
            return DbOperationResult(success=False, message=str(e))

    def search(self, text: str | List[str] | None = None, text_embedded: List[float] | List[List[float]] | None = None, n_results: int = 3) -> DbOperationResult:
        """Searches the collection for similar vectors."""
        if not self.client:
            return DbOperationResult(success=False, message="Client not connected")
        
        if text_embedded is None:
            self.logger.error("WeaviateDbRepository.search requires pre-computed embeddings.")
            return DbOperationResult(success=False, message="This repository requires the 'text_embedded' field.")

        try:
            query_vector = text_embedded
            
            response = self.client.collections.get(self.collection_name).query.near_vector(
                near_vector=query_vector,
                limit=n_results,
                return_metadata=weaviate.classes.query.MetadataQuery(distance=True),
                return_properties=["text_content", "metadata_json"]
            )
            
            documents = []
            for obj in response.objects:
                print(obj)
                props = obj.properties
                text = props.get("text_content", "")
                
                try:
                    metadata = json.loads(props.get("metadata_json", "{}"))
                except json.JSONDecodeError:
                    metadata = {}
                
                metadata['text'] = text
                metadata['distance'] = obj
                
                documents.append(MyDocument(
                    id=str(obj.uuid),
                    text=text,
                    embedding=[],
                    metadata=metadata
                ))
            
            return DbOperationResult(success=True, data=documents)
        except Exception as e:
            self.logger.error(f"Failed to search collection: {e}")
            traceback.print_exc()
            return DbOperationResult(success=False, message=str(e))

    def get_count(self) -> int:
        """Gets the exact count of objects in the collection."""
        if not self.client:
            self.logger.error("Client not connected, returning count 0")
            return 0
        try:
            result = self.client.collections.get(self.collection_name).aggregate.over_all(total_count=True)
            return result.total_count
        except Exception as e:
            # This can fail if the collection doesn't exist yet
            self.logger.warning(f"Failed to get count (collection might not exist): {e}")
            return 0

    def check_if_data_were_inserted(self) -> DbOperationResult:
        """Checks if the count is greater than 0."""
        count = self.get_count()
        if count > 0:
            return DbOperationResult(success=True, data={'count': count})
        else:
            return DbOperationResult(success=False, message="No data found.", data={'count': 0})

    def insert(self, docs: List[MyDocument]) -> DbOperationResult:
        """Inserts a list of MyDocument objects into the collection using a batch."""
        if not self.client:
            return DbOperationResult(success=False, message="Client not connected")
        
        try:
            with self.client.batch.dynamic() as batch:
                for doc in docs:
                    if not doc.embedding:
                        self.logger.warning(f"Skipping doc {doc.id}, no embedding found.")
                        continue
                    
                    properties = {
                        "text_content": doc.text,
                        "metadata_json": json.dumps(doc.metadata)
                    }
                    
                    batch.add_object(
                        collection=self.collection_name,
                        properties=properties,
                        vector=doc.embedding,
                        uuid=doc.id
                    )
            
            if len(self.client.batch.failed_objects) > 0:
                self.logger.error(f"Errors encountered during batch insert: {self.client.batch.errors}")
                return DbOperationResult(success=False, message=str(self.client.batch.errors))

            return DbOperationResult(success=True)
        except Exception as e:
            self.logger.error(f"Failed to insert points: {e}")
            traceback.print_exc()
            return DbOperationResult(success=False, message=str(e))

    def delete(self, ids: List[str]) -> DbOperationResult:
        """Deletes objects from the collection by their IDs using a batch operation."""
        if not self.client:
            return DbOperationResult(success=False, message="Client not connected")
        try:
            # Use a 'where' filter to delete all matching IDs in one go

            response = self.client.batch.delete_objects(
                class_name=self.collection_name,
                where=Filter.by_id().contains_any(ids)
            )
            
            if response.results and response.results.failed > 0:
                self.logger.error(f"Failed to delete some objects: {response.results.objects}")
                return DbOperationResult(success=False, message=str(response.results.objects))

            return DbOperationResult(success=True)
        except Exception as e:
            self.logger.error(f"Failed to delete points: {e}")
            return DbOperationResult(success=False, message=str(e))

    def close(self) -> DbOperationResult:
        """Closes the client connection."""
        try:
            if self.client:
                self.client.close()
                self.logger.info("Weaviate connection closed.")
            return DbOperationResult(success=True)
        except Exception as e:
            return DbOperationResult(success=False, message=str(e))