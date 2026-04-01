import os
import re

from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama

from rag.agents.state import AgentState
from rag.agents.enums import NodeName
from rag.agents.nodes.general_nodes import GeneralNodes
from rag.agents.nodes.rag_nodes import RagNodes

from database.base.base_db_repository import BaseDbRepository
from database.duck_db_repository import DuckDbRepository
from text_embedding import TextEmbeddingService
from utils.logging_config import get_logger

logger = get_logger(__name__)


def _context_window_from_model(model_name: str) -> int:
    match = re.search(r'(\d+(?:\.\d+)?)b', model_name.lower())
    size_b = float(match.group(1)) if match else None
    if size_b is None:
        return 8192
    if size_b < 4:
        return 4096
    if size_b < 12:
        return 8192
    if size_b <= 25:
        return 32768
    return 131072


def build_graph(
    database_service: BaseDbRepository,
    embedding_service: TextEmbeddingService,
    model_name: str = "ministral-3:8b",
    duck_db_repo: DuckDbRepository | None = None,
):
    context_window = _context_window_from_model(model_name)
    logger.info(f"Model '{model_name}' → context_window={context_window}")

    llm = ChatOllama(
        model=model_name,
        base_url=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        temperature=0,
        num_ctx=context_window,
        keep_alive='5m',
        repeat_penalty=1.0,
        timeout=120,
    )

    available_sources = database_service.get_all_filenames()
    router_nodes = GeneralNodes(llm, available_sources=available_sources)
    rag_nodes = RagNodes(
        llm,
        database_service,
        embedding_service,
        duck_db_repo=duck_db_repo,
        context_window=context_window,
        available_sources=available_sources,
    )

    builder = StateGraph(AgentState)
    # general
    builder.add_node(NodeName.ROUTER, router_nodes.router_agent)
    builder.add_node(NodeName.GENERAL, router_nodes.general_agent)
    # rag
    builder.add_node(NodeName.QUERY_PLANNER, rag_nodes.query_planner_agent)
    builder.add_node(NodeName.ANALYTICAL_QUERY, rag_nodes.analytical_query_agent)
    builder.add_node(NodeName.RESEARCH_WORKER, rag_nodes.research_worker)
    builder.add_node(NodeName.RETRIEVAL_GRADER_AGENT, rag_nodes.retrieval_grader_agent)
    builder.add_node(NodeName.FACT_EXTRACTOR, rag_nodes.fact_extractor_agent)
    builder.add_node(NodeName.SYNTHESIZER, rag_nodes.synthesizer_agent)
    builder.add_node(NodeName.COMPLETENESS_CHECKER, rag_nodes.completeness_checker_agent)
    builder.add_node(NodeName.HALLUCINATION_GRADER_AGENT, rag_nodes.hallucination_grader_agent)
    builder.add_node(NodeName.SCROLL_RETRIEVER, rag_nodes.scroll_retriever)
    builder.add_node(NodeName.ERROR, rag_nodes.error_agent)

    builder.add_edge(START, NodeName.ROUTER)

    builder.add_conditional_edges(
        NodeName.ROUTER,
        GeneralNodes.route_intent,
        path_map={
            NodeName.QUERY_PLANNER: NodeName.QUERY_PLANNER,
            NodeName.GENERAL: NodeName.GENERAL,
        }
    )
    builder.add_edge(NodeName.GENERAL, END)

    builder.add_conditional_edges(
        NodeName.QUERY_PLANNER,
        RagNodes.route_query_plan,
        path_map={
            NodeName.ANALYTICAL_QUERY: NodeName.ANALYTICAL_QUERY,
            NodeName.RESEARCH_WORKER: NodeName.RESEARCH_WORKER,
            NodeName.SCROLL_RETRIEVER: NodeName.SCROLL_RETRIEVER,
            NodeName.ERROR: NodeName.ERROR,
        }
    )
    builder.add_edge(NodeName.ERROR, END)

    builder.add_conditional_edges(
        NodeName.ANALYTICAL_QUERY,
        RagNodes.route_after_analytical,
        path_map={
            NodeName.RESEARCH_WORKER: NodeName.RESEARCH_WORKER,
            NodeName.SYNTHESIZER: NodeName.SYNTHESIZER,
        }
    )

    builder.add_edge(NodeName.SCROLL_RETRIEVER, NodeName.FACT_EXTRACTOR)

    builder.add_edge(NodeName.RESEARCH_WORKER, NodeName.RETRIEVAL_GRADER_AGENT)
    builder.add_edge(NodeName.RETRIEVAL_GRADER_AGENT, NodeName.FACT_EXTRACTOR)
    builder.add_edge(NodeName.FACT_EXTRACTOR, NodeName.SYNTHESIZER)
    builder.add_edge(NodeName.SYNTHESIZER, NodeName.COMPLETENESS_CHECKER)
    builder.add_conditional_edges(
        NodeName.COMPLETENESS_CHECKER,
        rag_nodes.route_completeness_check,
        path_map={
            NodeName.HALLUCINATION_GRADER_AGENT: NodeName.HALLUCINATION_GRADER_AGENT,
            NodeName.RESEARCH_WORKER: NodeName.RESEARCH_WORKER,
        }
    )
    builder.add_conditional_edges(
        NodeName.HALLUCINATION_GRADER_AGENT,
        rag_nodes.route_hallucination,
        path_map={
            NodeName.SYNTHESIZER: NodeName.SYNTHESIZER,
            END: END,
        }
    )

    return builder.compile()
