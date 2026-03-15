from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class ExtractionResult(BaseModel):
    """
    Structured output for metadata extraction from text chunks.
    
    Used with LLM structured output to ensure consistent formatting.
    """
    years: List[int] = Field(
        default_factory=list, 
        description='List of 4-digit years (1900-2100) mentioned in the text.'
    )
    locations: List[str] = Field(
        default_factory=list, 
        description='List of geographic locations: countries, regions, cities, continents, oceans.'
    )    
    entities: List[str] = Field(
        default_factory=list,
        description='List of organizations, institutions, or notable entities (e.g., WMO, IPCC, UN, NASA).'
    )    
    @field_validator('years', mode='before')
    @classmethod
    def coerce_years(cls, v):
        """Ensure years are integers."""
        if not v:
            return []
        result = []
        for item in v:
            try:
                year = int(item)
                result.append(year)
            except (ValueError, TypeError):
                continue
        return result
    
    @field_validator('locations', mode='before')
    @classmethod
    def clean_locations(cls, v):
        """Ensure locations are non-empty strings."""
        if not v:
            return []
        return [str(loc).strip() for loc in v if loc and str(loc).strip()]
    
    @field_validator('entities', mode='before')
    @classmethod
    def clean_entities(cls, v):
        """Ensure entities are non-empty strings."""
        if not v:
            return []
        return [str(ent).strip() for ent in v if ent and str(ent).strip()]