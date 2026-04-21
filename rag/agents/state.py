import operator
from langgraph.graph import add_messages
from typing import Annotated, List, Any
from typing_extensions import TypedDict

from database.base.my_document import MyDocument
from rag.agents.enums import Intent
from rag.agents.models import QueryPlan

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    intent: Intent | None
    detected_source: str | None
    rewritten_queries: List[str]
    query_plan: QueryPlan | None
    sql_result: str | None
    search_results: Annotated[list, operator.add]
    filtered_results: Annotated[List[MyDocument], operator.add]
    distilled_facts: Annotated[List[str], operator.add]
    hallucination_status: str
    hallucination_retries: int
    retrieval_iterations: int
    completeness_follow_up_query: str
    scientific_results: Annotated[List[Any], operator.add]
    data_source_scope: str | None

class WorkerState(TypedDict):
    query: str
    seen_doc_ids: List[str]