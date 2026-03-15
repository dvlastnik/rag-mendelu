import traceback
from typing import List, Dict, Any, Set, Optional
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import ast
import difflib

from database.base.MyDocument import MyDocument
from database.base.DbOperationResult import DbOperationResult
from database.base.BaseDbRepository import BaseDbRepository
from utils.logging_config import highlight_log

# Fields to create payload indexes on (improves filter performance)
# Only index fields you actually filter on - more indexes = more memory
INDEXED_FIELDS = {
    'source': 'keyword',           # Filter by source file
    'content_years': 'integer',    # Filter by year (stored as list)
    'locations': 'keyword',        # Filter by location (stored as list)
    'document_type': 'keyword',    # Filter by document type
    'has_numerical_data': 'bool',  # Filter for data-heavy chunks
}

# Fields to extract unique values for (for UI dropdowns, validation)
METADATA_FACET_FIELDS = ['years', 'locations', 'source']

# Hybrid search configuration
DEFAULT_SPARSE_WEIGHT = 0.3  # Weight for SPLADE (lexical)
DEFAULT_DENSE_WEIGHT = 0.7   # Weight for dense embeddings (semantic)

class QdrantDbRepository(BaseDbRepository):
    name = 'qdrant'

    def __init__(
        self, 
        ip: str, 
        port: int, 
        grpc_port: int = 6334, 
        collection_name: str = "", 
        metadata: Optional[Dict] = None
    ):
        """
        Initializes the Qdrant repository.
        
        Args:
            ip: Host IP for Qdrant.
            port: HTTP port for Qdrant (e.g., 6333).
            grpc_port: gRPC port for Qdrant (e.g., 6334).
            collection_name: Name of the collection.
            metadata: Must contain {'vector_size': int, 'distance': str}
                      e.g., {'vector_size': 768, 'distance': 'COSINE'}
        """
        metadata = metadata or {}
        super().__init__(ip, port, collection_name, metadata)
        self.client: Optional[QdrantClient] = None
        self.grpc_port = grpc_port
        
        self.vector_size = self.metadata.get('vector_size')
        if self.vector_size is None or not isinstance(self.vector_size, int):
            raise ValueError("QdrantDbRepository 'metadata' must include 'vector_size' (e.g., {'vector_size': 768})")
            
        distance_str = self.metadata.get('distance', 'DOT').upper()
        self.distance = getattr(models.Distance, distance_str, models.Distance.DOT)
        
        # Lazy-loaded metadata cache (populated on first access)
        self._valid_metadata_cache: Optional[Dict[str, List]] = None
        self._metadata_cache_dirty = True
        
        self.logger.info(f"QdrantRepository configured for collection '{collection_name}' with size {self.vector_size} and distance {self.distance}")

    @property
    def valid_metadata(self) -> Dict[str, List]:
        """Lazy-loaded metadata cache for filter validation."""
        if self._valid_metadata_cache is None or self._metadata_cache_dirty:
            self._valid_metadata_cache = self._get_unique_metadata_values(METADATA_FACET_FIELDS)
            self._metadata_cache_dirty = False
        return self._valid_metadata_cache
    
    def invalidate_metadata_cache(self) -> None:
        """Call after inserting new documents to refresh metadata cache."""
        self._metadata_cache_dirty = True

    def connect(self) -> DbOperationResult:
        """Connects to the Qdrant client and verifies connection."""
        try:
            self.logger.info(f'Connecting to Qdrant at {self.ip}:{self.port}...')

            self.client = QdrantClient(
                host=self.ip, 
                port=self.port, 
                grpc_port=self.grpc_port,
                timeout=30
            )
            
            # Verify connection with a simple operation
            collections = self.client.get_collections()
            self.logger.info(f'Qdrant connection successful. Found {len(collections.collections)} collections.')
            
            return DbOperationResult(success=True)
        except Exception as e:
            self.logger.error(f'Failed to connect to Qdrant: {e}')
            return DbOperationResult(success=False, message=str(e))

    def if_collection_exist_delete(self) -> DbOperationResult:
        """Checks if the collection exists and deletes it if it does."""
        if not self.client:
            return DbOperationResult(success=False, message='Client not connected')
        try:
            if self.client.collection_exists(self.collection_name):
                self.logger.warning(f"Collection '{self.collection_name}' already exists. Deleting.")
                self.client.delete_collection(self.collection_name)
            return DbOperationResult(success=True)
        except Exception as e:
            self.logger.error(f"Failed to delete collection: {e}")
            return DbOperationResult(success=False, message=str(e))

    def create_collection(self) -> DbOperationResult:
        """Creates a new collection with vector config and payload indexes."""
        if not self.client:
            return DbOperationResult(success=False, message='Client not connected')
        try:
            self.logger.info(f"Creating collection '{self.collection_name}'...")

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    'dense': models.VectorParams(
                        size=self.vector_size,
                        distance=self.distance
                    )
                },
                sparse_vectors_config={
                    'sparse': models.SparseVectorParams(
                        index=models.SparseIndexParams(
                            on_disk=True
                        )
                    )
                },
                # Optimize for filtering
                optimizers_config=models.OptimizersConfigDiff(
                    indexing_threshold=10000,  # Start indexing after 10k points
                ),
            )
            
            # Create payload indexes for filterable fields
            self._create_payload_indexes()
            
            self.logger.info('Collection created successfully with payload indexes.')
            return DbOperationResult(success=True)
        except Exception as e:
            self.logger.error(f'Failed to create collection: {e}')
            return DbOperationResult(success=False, message=str(e))
    
    def _create_payload_indexes(self) -> None:
        """
        Creates payload indexes for filterable fields.
        Indexes dramatically improve filter query performance.
        """
        if not self.client:
            return
            
        for field_name, field_type in INDEXED_FIELDS.items():
            try:
                if field_type == 'keyword':
                    schema = models.PayloadSchemaType.KEYWORD
                elif field_type == 'integer':
                    schema = models.PayloadSchemaType.INTEGER
                elif field_type == 'bool':
                    schema = models.PayloadSchemaType.BOOL
                else:
                    schema = models.PayloadSchemaType.KEYWORD
                
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=schema,
                )
                self.logger.debug(f"Created index for field '{field_name}' ({field_type})")
            except Exception as e:
                # Index might already exist
                self.logger.debug(f"Could not create index for '{field_name}': {e}")

    def _build_qdrant_filter(
        self, 
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> Optional[models.Filter]:
        """
        Converts a filter dictionary into a Qdrant Filter object.
        
        Handles two query patterns:
        1. Exact match: {'source': 'report.md'} -> source == 'report.md'
        2. List contains: {'content_years': 2022} -> 2022 in content_years[]
        3. Any of values: {'content_years': [2021, 2022]} -> any of [2021, 2022] in content_years[]
        
        Args:
            filter_dict: Dictionary of field->value filters.
            
        Returns:
            Qdrant Filter object or None if no valid filters.
        """
        if not filter_dict:
            return None
        
        # Fields that are stored as arrays in Qdrant
        array_fields = {'content_years', 'years', 'locations', 'topics', 'entities', 'headers'}
        
        must_conditions = []
        for key, value in filter_dict.items():
            if value in [None, '', []]:
                continue
            
            # Normalize to list for consistent handling
            values = value if isinstance(value, list) else [value]
            
            if key in array_fields:
                # For array fields: check if ANY of the query values exist in the stored array
                # E.g., content_years=[2021,2022,2023], query=2022 -> match
                must_conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchAny(any=values)
                    )
                )
            elif len(values) > 1:
                # Multiple values for non-array field: match any
                must_conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchAny(any=values)
                    )
                )
            else:
                # Single value exact match
                must_conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=values[0])
                    )
                )
        
        if not must_conditions:
            return None
            
        return models.Filter(must=must_conditions)
    

    def _get_unique_metadata_values(self, target_fields: List[str]) -> Dict[str, List[str]]:
        unique_data: Dict[str, Set[Any]] = {field: set() for field in target_fields}
        next_offset = None
        while True:
            records, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                with_payload=target_fields,
                with_vectors=False,
                limit=100,
                offset=next_offset
            )
            
            for record in records:
                if not record.payload:
                    continue
                    
                for field in target_fields:
                    raw_value = record.payload.get(field)
                    if raw_value is None:
                        continue

                    items_to_add = []

                    if isinstance(raw_value, str) and raw_value.strip().startswith('[') and raw_value.strip().endswith(']'):
                        try:
                            parsed = ast.literal_eval(raw_value)
                            if isinstance(parsed, list):
                                items_to_add = parsed
                            else:
                                items_to_add = [parsed]
                        except (ValueError, SyntaxError):
                            items_to_add = [raw_value]
                    elif isinstance(raw_value, list):
                        items_to_add = raw_value
                    else:
                        items_to_add = [raw_value]

                    for item in items_to_add:
                        if field == 'years':
                            try:
                                unique_data[field].add(int(item))
                            except (ValueError, TypeError):
                                unique_data[field].add(item)
                        else:
                            unique_data[field].add(item)
            
            if next_offset is None:
                break

        result = {}
        for field, values in unique_data.items():
            try:
                result[field] = sorted(list(values))
            except TypeError:
                result[field] = sorted(list(values), key=lambda x: str(x))
                
        return result

    def search(
        self, 
        text: Optional[str] = None,
        text_embedded: Optional[List[float]] = None,
        sparse_embedded: Any = None,
        filter_dict: Optional[Dict[str, Any]] = None,
        n_results: int = 3,
        prefetch_multiplier: int = 3,
        score_threshold: Optional[float] = None,
    ) -> DbOperationResult:
        """
        Performs Hybrid Search using dense + sparse (SPLADE) vectors with RRF fusion.
        
        Args:
            text: Original query text (for logging/debugging).
            text_embedded: Dense embedding vector for the query.
            sparse_embedded: Sparse (SPLADE) embedding for the query.
            filter_dict: Metadata filters to apply.
            n_results: Number of results to return.
            prefetch_multiplier: How many candidates to fetch before fusion (multiplier of n_results).
            score_threshold: Minimum score threshold (note: RRF scores are not normalized).
            
        Returns:
            DbOperationResult with list of MyDocument objects.
        """
        if not self.client:
            return DbOperationResult(success=False, message='Client not connected')
        
        if text_embedded is None:
            return DbOperationResult(success=False, message="Search requires 'text_embedded' (dense vector).")

        highlight_log(self.logger, 'Qdrant Hybrid Search')
        if text:
            self.logger.info(f'Query: "{text[:100]}"')
        self.logger.info(f'Filters: {filter_dict}')
        self.logger.debug(f'Dense dim: {len(text_embedded)}, Sparse terms: {len(sparse_embedded.indices) if sparse_embedded else 0}')

        try:
            q_filter = self._build_qdrant_filter(filter_dict)
            prefetch_limit = n_results * prefetch_multiplier

            # Hybrid search with RRF fusion
            if sparse_embedded:
                search_result = self.client.query_points(
                    collection_name=self.collection_name,
                    prefetch=[
                        # Sparse (SPLADE) - good for exact term matching
                        models.Prefetch(
                            query=models.SparseVector(
                                indices=sparse_embedded.indices,
                                values=sparse_embedded.values
                            ),
                            using='sparse',
                            filter=q_filter,
                            limit=prefetch_limit,
                        ),
                        # Dense - good for semantic similarity
                        models.Prefetch(
                            query=text_embedded,
                            using='dense',
                            filter=q_filter,
                            limit=prefetch_limit,
                        ),
                    ],
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    limit=n_results,
                    with_payload=True,
                    score_threshold=score_threshold,
                )
            else:
                # Dense-only fallback
                search_result = self.client.query_points(
                    collection_name=self.collection_name,
                    query=text_embedded,
                    using='dense',
                    query_filter=q_filter,
                    limit=n_results,
                    with_payload=True,
                    score_threshold=score_threshold,
                )
            
            result_count = len(search_result.points)
            self.logger.info(f'Results: {result_count} documents')
            
            if result_count == 0:
                self.logger.warning('⚠️ No documents found. Consider relaxing filters.')
                if filter_dict:
                    self.logger.warning(f'   Active filters: {filter_dict}')

            documents = self._points_to_documents(search_result.points)
            
            highlight_log(self.logger, 'Qdrant Hybrid Search', only_char=True)
            return DbOperationResult(success=True, data=documents)
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            traceback.print_exc()
            return DbOperationResult(success=False, message=str(e))
    
    def _points_to_documents(self, points) -> List[MyDocument]:
        """
        Converts Qdrant search results to MyDocument objects.
        
        Note: We extract 'text' from payload since it's stored there,
        but ideally text should be stored separately or in a document store.
        """
        documents = []
        for point in points:
            payload = point.payload or {}
            
            # Get text from payload (consider: separate document store for large texts)
            text = payload.get('text', '') or payload.get('parent_text', '')
            
            # Build clean metadata without redundant text field
            metadata = {k: v for k, v in payload.items() if k not in ('text',)}
            metadata['score'] = point.score
            metadata['point_id'] = point.id
            
            documents.append(
                MyDocument(
                    id=str(point.id),
                    text=text,
                    embedding=[],  # Don't return embeddings (save memory)
                    metadata=metadata,
                    score=point.score
                )
            )
        return documents

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
        
    def validate_filter(self, value, key_name: str, fuzzy_cutoff: float = 0.85):
        """
        Validates if a filter value exists in the database, with optional fuzzy matching.
        
        Args:
            value: The value to validate (e.g., 'pakistan', 2022)
            key_name: The field name to validate against (e.g., 'locations', 'years')
            fuzzy_cutoff: Similarity threshold for fuzzy matching (0.0-1.0)
            
        Returns:
            The validated value (possibly fuzzy-matched), or None if not found.
        """
        result = None
        
        if isinstance(value, str):
            value = value.lower()
        
        if not value:
            return None
        
        # Try exact match first
        count_result = self.client.count(
            collection_name=self.collection_name,
            count_filter=models.Filter(
                must=[models.FieldCondition(key=key_name, match=models.MatchValue(value=value))]
            )
        )
        
        if count_result.count > 0:
            result = value
            self.logger.info(f"✅ Filter Validated: {key_name} '{value}' has {count_result.count} docs.")
        else:
            # Try fuzzy matching for string values
            if isinstance(value, str):
                all_values = self.valid_metadata.get(key_name, [])
                if all_values:
                    # Convert to lowercase for comparison
                    all_values_lower = [str(v).lower() for v in all_values]
                    matches = difflib.get_close_matches(value, all_values_lower, n=1, cutoff=fuzzy_cutoff)
                    
                    if matches:
                        # Find the original case-preserved value
                        matched_lower = matches[0]
                        original_value = None
                        for orig, lower in zip(all_values, all_values_lower):
                            if lower == matched_lower:
                                original_value = orig if not isinstance(orig, str) else orig.lower()
                                break
                        
                        if original_value:
                            # Verify the fuzzy match exists in DB
                            verify_count = self.client.count(
                                collection_name=self.collection_name,
                                count_filter=models.Filter(
                                    must=[models.FieldCondition(key=key_name, match=models.MatchValue(value=original_value))]
                                )
                            )
                            if verify_count.count > 0:
                                result = original_value
                                self.logger.info(f"✅ Fuzzy Match: {key_name} '{value}' → '{original_value}' ({verify_count.count} docs)")
                            else:
                                self.logger.warning(f"⚠️ Filter Dropped: {key_name} '{value}' fuzzy-matched to '{original_value}' but has 0 docs.")
                        else:
                            self.logger.warning(f"⚠️ Filter Dropped: {key_name} '{value}' no valid fuzzy match found.")
                    else:
                        self.logger.warning(f"⚠️ Filter Dropped: {key_name} '{value}' has 0 docs (no fuzzy match above {fuzzy_cutoff}).")
                else:
                    self.logger.warning(f"⚠️ Filter Dropped: {key_name} '{value}' has 0 docs (no metadata available).")
            else:
                self.logger.warning(f"⚠️ Filter Dropped: {key_name} '{value}' has 0 docs.")
        
        return result
        
    def get_all_filenames(self) -> List[str]:
        """
        Gets all source files that are ingested in current database.
        """
        unique_filenames = set()
        next_offset = None
        
        while True:
            records, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                with_payload=['source'],
                with_vectors=False,
                limit=100,
                offset=next_offset
            )
            
            for record in records:
                if record.payload:
                    source = record.payload.get('source')
                    if source:
                        unique_filenames.add(source)
            
            if next_offset is None:
                break
                
        return sorted(list(unique_filenames))

    def check_if_data_were_inserted(self) -> DbOperationResult:
        """Checks if the count is greater than 0."""
        count = self.get_count()
        if count > 0:
            return DbOperationResult(success=True, data=count)
        else:
            return DbOperationResult(success=False, message="No data found.", data=0)

    def insert(self, docs: List[MyDocument], batch_size: int = 100) -> DbOperationResult:
        """
        Inserts a list of MyDocument objects into the collection.
        
        Args:
            docs: List of documents to insert.
            batch_size: Number of points per upsert batch (for large inserts).
            
        Returns:
            DbOperationResult with success status.
        """
        if not self.client:
            return DbOperationResult(success=False, message="Client not connected")
        
        points = []
        skipped = 0
        
        for doc in docs:
            if not doc.embedding:
                self.logger.warning(f"Skipping doc {doc.id}, no embedding found.")
                skipped += 1
                continue

            vector_payload = {"dense": doc.embedding}

            if doc.sparse_embedding:
                vector_payload["sparse"] = models.SparseVector(
                    indices=doc.sparse_embedding.indices,
                    values=doc.sparse_embedding.values
                )
            
            # Ensure metadata includes text for retrieval
            payload = doc.metadata.copy() if doc.metadata else {}
            if 'text' not in payload and doc.text:
                payload['text'] = doc.text
            
            point = models.PointStruct(
                id=str(doc.id),
                vector=vector_payload,
                payload=payload
            )
            points.append(point)
            
        if not points:
            return DbOperationResult(success=False, message="No valid documents with embeddings to insert.")
        
        self.logger.info(f"Inserting {len(points)} points ({skipped} skipped)...")
            
        try:
            # Batch insert for large document sets
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch,
                    wait=True
                )
                if len(points) > batch_size:
                    self.logger.debug(f"Inserted batch {i // batch_size + 1}/{(len(points) - 1) // batch_size + 1}")
            
            # Invalidate metadata cache since we added new documents
            self.invalidate_metadata_cache()
            
            self.logger.info(f"Successfully inserted {len(points)} points.")
            return DbOperationResult(success=True, data={'inserted': len(points), 'skipped': skipped})
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