from pydantic import BaseModel, Field
from typing import List

from rag.agents.enums import Intent

class MultiQuery(BaseModel):
    queries: List[str] = Field(
        ...,
        description="2–5 targeted search queries covering all aspects of the user question."
    )

class ExtractionScheme(BaseModel):
    source: str | None = None

class GeneralOrRagDecision(BaseModel):
    intent: Intent = Field(
        ...,
        description="Is the user asking for data (rag), listing all items (listing), or just chatting (general)?"
    )
    detected_source: str | None = Field(
        default=None,
        description="The source/dataset name the user is asking about, when intent is 'listing'."
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

class CompletenessCheck(BaseModel):
    is_complete: bool = Field(
        ...,
        description="True if the answer fully addresses the question."
    )
    follow_up_query: str = Field(
        ...,
        description="2-5 keywords for missing info. Empty if complete."
    )