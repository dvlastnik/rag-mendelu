import uuid
from typing import List

from text_embedding.base import BaseDenseEmbeddingLibrary
from text_embedding.models import EmbeddingResponse
from text_embedding.sparse_library import SparseEmbeddingLibrary
from utils.Utils import Utils
from utils.logging_config import get_logger

logger = get_logger(__name__)

LIBRARIES = ("fastembed", "sentence_transformers")


class TextEmbeddingService:
    def __init__(
        self,
        library: str = "fastembed",
        dense_model: str | None = None,
        sparse_model: str = "prithivida/Splade_PP_en_v1",
    ):
        self._dense_library: BaseDenseEmbeddingLibrary = self._init_dense_library(library, dense_model)
        self._sparse_library = SparseEmbeddingLibrary(model_name=sparse_model)
        self._library_name = library

    def get_embedding_with_uuid(
        self,
        data: List[str] | str,
        chunk_size: int | None = None,
    ) -> List[EmbeddingResponse]:
        texts = [data] if isinstance(data, str) else data

        if not chunk_size:
            return self._embed_batch(texts)

        result: List[EmbeddingResponse] = []
        for i, chunk in enumerate(Utils.chunks(texts, chunk_size)):
            try:
                logger.debug(f"- {i}. chunk processed")
                result.extend(self._embed_batch(chunk))
            except Exception as e:
                logger.error(f"Batch failed: {e}")
        return result

    def set_model(self, model_name: str):
        self._dense_library.set_model(model_name)

    def set_library(self, library: str):
        if library not in LIBRARIES:
            raise ValueError(f"Unknown library '{library}'. Choose from: {LIBRARIES}")
        self._dense_library = self._init_dense_library(library, dense_model=None)
        self._library_name = library

    def get_current_model(self) -> str:
        return self._dense_library.get_current_model()

    def get_library(self) -> str:
        return self._library_name

    def _embed_batch(self, texts: List[str]) -> List[EmbeddingResponse]:
        dense_vectors = self._dense_library.encode(texts)
        sparse_vectors = self._sparse_library.embed(texts)

        return [
            EmbeddingResponse(
                uuid=str(uuid.uuid4()),
                embedding=dense,
                sparse=sparse,
            )
            for dense, sparse in zip(dense_vectors, sparse_vectors)
        ]

    @staticmethod
    def _init_dense_library(library: str, dense_model: str | None) -> BaseDenseEmbeddingLibrary:
        if library == "fastembed":
            from text_embedding.fastembed_library import FastEmbedLibrary, DEFAULT_MODEL
            return FastEmbedLibrary(model_name=dense_model or DEFAULT_MODEL)
        if library == "sentence_transformers":
            from text_embedding.sentence_transformers_library import SentenceTransformersLibrary, DEFAULT_MODEL
            return SentenceTransformersLibrary(model_name=dense_model or DEFAULT_MODEL)
        raise ValueError(f"Unknown library '{library}'. Choose from: {LIBRARIES}")
