import operator
from langgraph.graph import add_messages
from typing import Annotated, List
from typing_extensions import TypedDict

from rag.agents.enums import Intent
from rag.agents.models import ExtractionScheme

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    extracted_data: List[ExtractionScheme]
    intent: Intent | None
    search_results: Annotated[list, operator.add]
    filtered_results: List[str]
    rewritten_query: str
    hallucination_status: str
    hallucination_retries: int
    
class WorkerState(TypedDict):
    target: ExtractionScheme
    query: str