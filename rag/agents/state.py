import operator
from langgraph.graph import add_messages
from typing import Annotated, List
from typing_extensions import TypedDict

from database.base.MyDocument import MyDocument
from rag.agents.enums import Intent
from rag.agents.models import ExtractionScheme

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    extracted_data: List[ExtractionScheme]
    intent: Intent | None
    detected_source: str | None
    rewritten_queries: List[str]
    search_results: Annotated[list, operator.add]
    filtered_results: List[MyDocument]
    distilled_facts: Annotated[List[str], operator.add]
    hallucination_status: str
    hallucination_retries: int
    retrieval_iterations: int
    completeness_follow_up_query: str

class WorkerState(TypedDict):
    target: ExtractionScheme
    query: str
    seen_doc_ids: List[str]  # IDs already fetched in prior iterations; empty on first call