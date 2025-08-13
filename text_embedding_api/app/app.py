from fastapi import FastAPI, HTTPException, Depends, Query
import uuid
import sentence_transformers

from models import *
from EmbeddingServiceManager import EmbeddingServiceManager
from BaseEmbeddingModelService import BaseEmbeddingModelService

app = FastAPI()
embedding_manager = EmbeddingServiceManager()

def get_embedding_manager():
    return embedding_manager

def get_embedding_service():
    return embedding_manager.active_service

@app.get('/')
async def root():
    return 'This server only to achieve text embeddings from various embedding models.'

@app.post(
    path='/set-embedding-service',
    responses={400: {"description": f"Bad Request - This service name is not known"}}
)
async def set_embedding_service(service_name: str = Query(..., description="Name of the embedding service to activate")):
    try:
        embedding_manager.set_active_service(service_name)
    except ValueError:
        raise HTTPException(status_code=400, detail={"description": f"Bad Request - This service name is not known"})
    return {"description": f"Active embedding service set to '{service_name}'"}

@app.post(
    path='/embed-text', 
    response_model=EmbedTextResponse, 
    responses={400: {"description": "Bad Request - No text provided"}}
)
async def embed_text(request: EmbedTextRequest, embed_model: BaseEmbeddingModelService = Depends(get_embedding_service)):
    if not request.texts:
        raise HTTPException(status_code=400, detail="No text to embed provided")
    
    texts = [t for t in request.texts if t.strip()]
    
    embeddings = embed_model.encode(texts)

    data = [
        EmbedText(uuid=str(uuid.uuid4()), embeddings=embedding)
        for embedding in embeddings
    ]

    return EmbedTextResponse(data=data)

@app.post(
    path='/change-embedding-model',
    responses={400: {"description": "Bad Request - No model_name provided"}}
)
async def change_embedding_model(request: ChangeEmbeddingModelRequest, embed_model: BaseEmbeddingModelService = Depends(get_embedding_service)):
    if not request.model_name:
        raise HTTPException(status_code=400, detail="No model name provided")
    
    try:
        embed_model.set_model(model_name=request.model_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error: " + str(e))

    return {"description": "Model changed successfully"}

@app.get(
    path='/get-available-models'
)
async def get_available_models(embed_model: BaseEmbeddingModelService = Depends(get_embedding_service)):
    return embed_model.get_installed_models()
        
