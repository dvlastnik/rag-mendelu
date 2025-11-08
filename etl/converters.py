from functools import wraps
from typing import Any, Callable, Dict
from pathlib import Path
from pandas import DataFrame, read_csv, read_excel
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader

from utils.logging_config import get_logger
from utils.Utils import Utils

logger = get_logger(__name__)

type ConvertFuncData = Dict[str, Any]
type ConvertFunc = Callable[[ConvertFuncData], DataFrame | None]

converter_functions: Dict[str,  ConvertFunc] = {}

def register_converter(name: str):
    def decorator(func: ConvertFunc):
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)
        
        converter_functions[name] = wrapper

    return decorator

@register_converter(".csv")
def csv_converter(file: Path) -> dict:
    return read_csv(file)

@register_converter(".xlsx")
def xlsx_convertor(file: Path):
    return read_excel(file)

@register_converter(".pdf")
def pdf_converter(file: Path) -> None:
    # loader = PyPDFLoader(file_path=file)
    # logger.info("Extracting...")
    # return loader.load()
    Utils.convert_pdf_to_md(file, output_folder=f"data/drough")

def convert_data(file: Path) -> DataFrame | None:
    converter = converter_functions.get(file.suffix.lower())
    if converter is None:
        raise ValueError(f"This ETL script is not implemented for this type of file: {file.suffix}")

    return converter(file)
