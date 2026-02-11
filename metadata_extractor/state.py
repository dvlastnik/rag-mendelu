from typing import TypedDict, Dict

from metadata_extractor.models import ExtractionResult

class ExtractorAgentState(TypedDict):
    text_chunk: str
    raw_extraction: ExtractionResult
    final_extraction: ExtractionResult
    clean_data: Dict