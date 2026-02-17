from typing import TypedDict, Dict, Optional, List

from metadata_extractor.models import ExtractionResult


class ExtractorAgentState(TypedDict, total=False):
    """
    State for the metadata extraction graph.
    
    Using total=False makes all fields optional, which matches
    the progressive population pattern in the graph.
    """
    # Input
    text_chunk: str
    
    # Intermediate (populated by extractor node)
    raw_extraction: Optional[ExtractionResult]
    extraction_error: Optional[str]
    
    # Output (populated by cleaning node)
    clean_data: Dict[str, List]  # Contains: years, locations, entities, has_numerical_data