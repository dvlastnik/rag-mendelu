from fastapi import FastAPI, HTTPException, Depends
import uuid
import sentence_transformers

from models import *
from EmbeddingModelService import EmbeddingModelService

app = FastAPI()

def get_embedding_model_service():
    return EmbeddingModelService()

@app.get('/')
async def root():
    return 'This server only to achieve text embeddings from various embedding models.'

@app.post(
    path='/embed-text', 
    response_model=EmbedTextResponse, 
    responses={400: {"description": "Bad Request - No text provided"}}
)
async def embed_text(request: EmbedTextRequest, embed_model: EmbeddingModelService = Depends(get_embedding_model_service)):
    if not request.texts:
        raise HTTPException(status_code=400, detail="No text to embed provided")
    
    data = []
    
    for text in request.texts:
        if text != '':
            data.append(
                EmbedText(
                    uuid=str(uuid.uuid4()),
                    embeddings=embed_model.encode(text)
                )
            )

    return EmbedTextResponse(data=data)

@app.post(
    path='/change-embedding-model',
    responses={400: {"description": "Bad Request - No model_name provided"}}
)
async def change_embedding_model(request: ChangeEmbeddingModelRequest, embed_model: EmbeddingModelService = Depends(get_embedding_model_service)):
    if not request.model_name:
        raise HTTPException(status_code=400, detail="No model name provided")
    
    try:
        embed_model.set_model(model_name=request.model_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error: " + str(e))

    return {"message": "Model changed successfully"}

@app.get(
    path='/get-available-models'
)
async def get_available_models(embed_model: EmbeddingModelService = Depends(get_embedding_model_service)):
    return embed_model.get_installed_models()
        
