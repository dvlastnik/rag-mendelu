import uuid
from typing import List, Union, Any

from models import (
    EmbedTextResponse, 
    EmbedTextRequest, 
    EmbedText, 
    ChunkAndEmbedResponse, 
    ChunkAndEmbed, 
    SparseVectorData
)
from embedding.BaseEmbeddingModelService import BaseEmbeddingModelService
from embedding.SparseEmbeddingService import SparseEmbeddingService

def embed_and_return_response(
        request_text: Union[EmbedTextRequest, List[str]], 
        embed_model: BaseEmbeddingModelService, 
        sparse_model: None | SparseEmbeddingService = None,
        with_original_text: bool = False
    ) -> Union[EmbedTextResponse, ChunkAndEmbedResponse]:
    
    if sparse_model is None and with_original_text:
        raise ValueError("Sparse Model is required for ChunkAndEmbed operations.")

    texts = request_text
    if isinstance(request_text, EmbedTextRequest):
        texts = [t for t in request_text.texts if t.strip()]
    
    if not texts:
        return ChunkAndEmbedResponse(data=[]) if with_original_text else EmbedTextResponse(data=[])

    try:
        sparse_results = list(sparse_model.embed(texts))
    except Exception as e:
        raise RuntimeError(f"Sparse Embedding failed: {e}")

    dense_embeddings = embed_model.encode(texts)
    if not with_original_text:
        print(sparse_results)
        data = [
            EmbedText(uuid=str(uuid.uuid4()), embeddings=embedding)
            for embedding in dense_embeddings
        ]
        return EmbedTextResponse(data=data, sparse_data=SparseVectorData(**sparse_results[0]))

    data = []
    for text, dense, sparse_dict in zip(texts, dense_embeddings, sparse_results):
        sparse_data = SparseVectorData(**sparse_dict)

        data.append(
            ChunkAndEmbed(
                text=text, 
                embed_text=EmbedText(
                    uuid=str(uuid.uuid4()), 
                    embeddings=dense
                ),
                sparse_embedding=sparse_data 
            )
        )

    return ChunkAndEmbedResponse(data=data)