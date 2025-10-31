from pydantic import BaseModel
from typing import List

# embed-text
class EmbedText(BaseModel):
    uuid: str
    embeddings: List[float]

class EmbedTextRequest(BaseModel):
    texts: List[str]

class EmbedTextResponse(BaseModel):
    data: List[EmbedText]

###########

# change-embedding-model
class ChangeEmbeddingModelRequest(BaseModel):
    model_name: str

###########

# get_available_models
class GetAvailableModelsResponse(BaseModel):
    model_names: List[str]
###########

# get-current-model
class GetCurrentModelResponse(BaseModel):
    embedding_model_name: str
    embedding_library_name: str
###########

# chunk similarity splitter
class ChunkBySimilarityRequest(BaseModel):
    text: str

class ChunkBySimilarityResponse(BaseModel):
    sentences: List[str]
###########

# Chunk and embed
class ChunkAndEmbed(BaseModel):
    text: str
    embed_text: EmbedText

class ChunkAndEmbedResponse(BaseModel):
    data: List[ChunkAndEmbed]
###########
