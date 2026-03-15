from pydantic import BaseModel, Field
from typing import List

from rag.agents.enums import Intent

class MultiQuery(BaseModel):
    queries: List[str] = Field(
        ...,
        description="Exactly 2 alternative search query rephrasings of the user question."
    )

class ExtractionScheme(BaseModel):
    location: str | None
    year: int | None
    entities: List[str] | None

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
        description="Documents are relevant to the user query, 'yes' or 'no'."
    )

class GradeDocumentsBatch(BaseModel):
    relevant_indices: List[int] = Field(
        ...,
        description="List of integer indices (0, 1, 2) of the documents that are relevant."
    )

class GradeHallucinations(BaseModel):
    is_relevant: str = Field(
        ...,
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

class GapCheck(BaseModel):
    is_sufficient: bool = Field(
        ...,
        description="True if the context fully answers the question; false if key facts are still missing."
    )
    follow_up_query: str = Field(
        ...,
        description="Focused keyword query targeting the specific missing information. Empty string if is_sufficient is true."
    )