from typing import List
from pydantic import BaseModel, Field

class ExtractionResult(BaseModel):
    """Aggregated lists of metadata found in the text."""
    years: List[int] = Field(default_factory=list, description='List of all 4-digit years found.')
    locations: List[str] = Field(default_factory=list, description='List of all mentioned location like cities, countries, regions or continents.')