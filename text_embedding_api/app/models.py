from pydantic import BaseModel

# embed-text
class EmbedText(BaseModel):
    uuid: str
    embeddings: list[float]

class EmbedTextRequest(BaseModel):
    texts: list[str]

class EmbedTextResponse(BaseModel):
    data: list[EmbedText]

###########

# change-embedding-model
class ChangeEmbeddingModelRequest(BaseModel):
    model_name: str

###########

