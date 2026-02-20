from functools import wraps
from typing import Any, Callable, Dict
from pathlib import Path
from pandas import DataFrame, read_csv, read_excel

from utils.logging_config import get_logger
from utils.Utils import Utils

logger = get_logger(__name__)

type ConvertFuncData = Dict[str, Any]
type ConvertFunc = Callable[[ConvertFuncData], DataFrame | None]

converter_functions: Dict[str, ConvertFunc] = {}

def register_converter(name: str):
    def decorator(func: ConvertFunc):
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        converter_functions[name] = wrapper

    return decorator

@register_converter(".pdf")
def pdf_converter(file: Path, output_folder: str = "data") -> None:
    Utils.convert_to_md(file, output_folder=output_folder)

@register_converter(".docx")
def docx_converter(file: Path, output_folder: str = "data") -> None:
    Utils.convert_to_md(file, output_folder=output_folder)

@register_converter(".pptx")
def pptx_converter(file: Path, output_folder: str = "data") -> None:
    Utils.convert_to_md(file, output_folder=output_folder)

@register_converter(".md")
def md_converter(file: Path, output_folder: str = "data") -> None:
    output_path = Utils.get_output_path(file, output_folder)
    if not output_path.exists():
        output_path.write_text(file.read_text(encoding="utf-8"), encoding="utf-8")

@register_converter(".txt")
def txt_converter(file: Path, output_folder: str = "data") -> None:
    output_path = Utils.get_output_path(file, output_folder)
    if not output_path.exists():
        output_path.write_text(file.read_text(encoding="utf-8"), encoding="utf-8")

@register_converter(".csv")
def csv_converter(file: Path, _output_folder: str = "data") -> DataFrame:
    return read_csv(file)

@register_converter(".xlsx")
def xlsx_converter(file: Path, _output_folder: str = "data") -> DataFrame:
    return read_excel(file)


def convert_data(file: Path, output_folder: str = "data") -> DataFrame | None:
    converter = converter_functions.get(file.suffix.lower())
    if converter is None:
        raise ValueError(f"This ETL script is not implemented for this type of file: {file.suffix}")

    return converter(file, output_folder)
