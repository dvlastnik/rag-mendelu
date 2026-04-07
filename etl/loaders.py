from functools import wraps
from typing import Any, Callable, Dict

from etl.base_etl import BaseEtl
from etl.etl_state import ETLState
from utils.logging_config import get_logger

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

def load_data(etl_instance: BaseEtl) -> ETLState:
    loader = loader_functions.get(etl_instance.file.suffix.lower())
    if loader is None:
        return etl_instance._insert_by_chunks()
    return loader(etl_instance)