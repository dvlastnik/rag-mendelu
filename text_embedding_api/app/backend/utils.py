from typing import List
import uuid

from models import EmbedTextResponse, EmbedTextRequest, EmbedText, ChunkAndEmbedResponse, ChunkAndEmbed
from embedding.BaseEmbeddingModelService import BaseEmbeddingModelService

def embed_and_return_response(request_text: EmbedTextRequest | List[str], embed_model: BaseEmbeddingModelService, with_original_text: bool = False) -> EmbedTextResponse | ChunkAndEmbedResponse:
    texts = request_text
    if isinstance(request_text, EmbedTextRequest):
        texts = [t for t in request_text.texts if t.strip()]

    embeddings = embed_model.encode(texts)

    if not with_original_text:
        data = [
            EmbedText(uuid=str(uuid.uuid4()), embeddings=embedding)
            for embedding in embeddings
        ]

        return EmbedTextResponse(data=data)
    
    data = [
        ChunkAndEmbed(
            text=text, 
            embed_text=EmbedText(uuid=str(uuid.uuid4()), embeddings=embedding)
        )
        for text, embedding in zip(texts, embeddings)
    ]

    return ChunkAndEmbedResponse(data=data)