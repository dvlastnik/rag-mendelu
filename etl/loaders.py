from functools import wraps
from typing import Any, Callable, Dict, List
from pathlib import Path
import pandas as pd

from utils.logging_config import get_logger
from etl.BaseEtl import BaseEtl
from etl.EtlState import ETLState
from database.base.MyDocument import MyDocument
from utils.Utils import Utils

logger = get_logger(__name__)

type LoaderFuncData = Dict[str, Any]
type LoaderFunc = Callable[[LoaderFuncData], ETLState]

loader_functions: Dict[str, LoaderFunc] = {}

def register_loader(name: str):
    def decorator(func: LoaderFunc):
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)
        
        loader_functions[name] = wrapper

    return decorator

@register_loader(".csv")
def csv_loader(etl_instance: BaseEtl) -> ETLState:
    etl_instance.documents = etl_instance.df.apply(etl_instance._row_to_document, axis=1).tolist()
    texts = [doc.text for doc in etl_instance.documents]

    embeddings_response = etl_instance.embedding_service.get_embedding_with_uuid(texts, chunk_size=200)
    for doc, embed_text in zip(etl_instance.documents, embeddings_response):
        doc.embedding = embed_text.embedding
        doc.id = embed_text.uuid
    
    return etl_instance._insert_by_chunks()

@register_loader(".pdf")
def pdf_loader(etl_instance: BaseEtl) -> ETLState:
    return etl_instance._insert_by_chunks()

def load_data(etl_instance: BaseEtl) -> ETLState:
    loader = loader_functions.get(etl_instance.file.suffix.lower())
    if loader is None:
        raise ValueError(f"This loader script is not implemented for this type of file: {etl_instance.file.suffix}")
    return loader(etl_instance)