from pydantic import BaseModel, Field
from typing import List

from rag.agents.enums import Intent

class ExtractionScheme(BaseModel):
    country: str | None
    city: str | None
    year: str | None
    topics: List[str]

class GeneralOrRagDecision(BaseModel):
    intent: Intent = Field(
        ..., 
        description="Is the user asking for data (rag) or just chatting (general)?"
    )

class MultiExtraction(BaseModel):
    targets: List[ExtractionScheme] = Field(
        ...,
        description="Extract all countries, years, cities and topics mentioned in the query."
    )

class GradeDocuments(BaseModel):
    is_relevant: str = Field(
        ...,
        description="Documents are relevant to the user query, 'yes' or 'no'"
    )

class GradeHallucinations(BaseModel):
    is_relevant: str = Field(
        ...,
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )