from enum import Enum
from pydantic import BaseModel, Field
from typing import List

from rag.agents.enums import Intent


class QueryStrategy(str, Enum):
    VECTOR = "vector"
    SQL = "sql"
    HYBRID = "hybrid"
    SCROLL = "scroll"

class MultiQuery(BaseModel):
    queries: List[str] = Field(
        ...,
        description="2–5 targeted search queries covering all aspects of the user question."
    )

class GeneralOrRagDecision(BaseModel):
    intent: Intent = Field(
        ...,
        description="Query type: 'rag' for specific factual questions, 'exhaustive' for listing/enumeration queries (list all, every, all X mentioned), 'summarization' for summarize/overview requests, 'general' for greetings only."
    )
    detected_source: str | None = Field(
        default=None,
        description="The source/dataset name the user is asking about, if identifiable. Must match one of the available sources."
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

class QueryPlan(BaseModel):
    strategy: QueryStrategy = Field(
        ...,
        description="Retrieval strategy: 'vector' for semantic search, 'sql' for aggregation/filter queries on tabular data, 'hybrid' for both SQL filtering and semantic search, 'scroll' for full-document retrieval."
    )
    vector_queries: List[str] = Field(
        default_factory=list,
        description="2–5 keyword-dense search queries for vector/hybrid strategies. Empty for sql/scroll."
    )
    sql_sources: List[str] = Field(
        default_factory=list,
        description="DuckDB table names to query. Required for sql/hybrid strategies. List 1 or more table names; multiple tables will be combined with UNION ALL."
    )
    sql_hint: str | None = Field(
        default=None,
        description="Natural language description of the analytical operation, e.g. 'find the row with maximum review score'."
    )

class SQLQueryPlan(BaseModel):
    sql: str = Field(..., description="A valid SELECT SQL statement to answer the question.")
    explanation: str = Field(..., description="One sentence explaining what this SQL computes.")