from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
import uuid

from models import *
from embedding.EmbeddingServiceManager import EmbeddingServiceManager
from embedding.BaseEmbeddingModelService import BaseEmbeddingModelService
from chunking.SentenceSimiliarity import SentenceSimilarity
from chunking.SimilarSentenceSplitter import SimilarSentenceSplitter

app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

embedding_manager = EmbeddingServiceManager()

sentence_similarity_model = SentenceSimilarity(embedding_service=embedding_manager.services["sentence_transformers"])
chunking_model = SimilarSentenceSplitter(similarity_model=sentence_similarity_model)

def get_embedding_manager():
    return embedding_manager

def get_embedding_service():
    return embedding_manager.active_service

def get_similarity_sentence_splitter():
    return chunking_model

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
    
    try:
        texts = [t for t in request.texts if t.strip()]
        
        embeddings = embed_model.encode(texts)

        data = [
            EmbedText(uuid=str(uuid.uuid4()), embeddings=embedding)
            for embedding in embeddings
        ]

        return EmbedTextResponse(data=data)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error: " + str(e))

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
    path='/get-available-models',
    response_model=GetAvailableModelsResponse
)
async def get_available_models(embed_model: BaseEmbeddingModelService = Depends(get_embedding_service)):
    return GetAvailableModelsResponse(model_names=embed_model.get_installed_models())

@app.get(
    path='/get-current-model',
    response_model=GetCurrentModelResponse
)
async def get_current_model(
    embed_model: BaseEmbeddingModelService = Depends(get_embedding_service), 
    embed_manager: EmbeddingServiceManager = Depends(get_embedding_manager)
):
    return GetCurrentModelResponse(
        embedding_model_name=embed_model.get_current_model(),
        embedding_library_name=embed_manager.active_service_name
    )

# TODO: Endpoint for chunking text
@app.post(
    path='/chunk-by-similarity',
    response_model=ChunkBySimilarityResponse,
    responses={400: {"description": "Bad Request - No text provided"}}
)
async def chunk_similarity_split(request: ChunkBySimilarityRequest, chunking_model: SimilarSentenceSplitter = Depends(get_similarity_sentence_splitter), embed_model: BaseEmbeddingModelService = Depends(get_embedding_service)):
    if not request.text:
        raise HTTPException(status_code=400, detail="No text provided")
    
    try:
        sentences = chunking_model.split_text(request.text)
        return ChunkBySimilarityResponse(sentences=sentences)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error: " + str(e))