import re

from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama

from rag.agents.state import AgentState
from rag.agents.enums import NodeName
from rag.agents.nodes.general_nodes import GeneralNodes
from rag.agents.nodes.rag_nodes import RagNodes

from database.base.BaseDbRepository import BaseDbRepository
from text_embedding import TextEmbeddingService
from utils.logging_config import get_logger

logger = get_logger(__name__)


def _context_window_from_model(model_name: str) -> int:
    match = re.search(r'(\d+(?:\.\d+)?)b', model_name.lower())
    size_b = float(match.group(1)) if match else None
    if size_b is None:
        return 8192
    if size_b < 4:
        return 4096    # 3b
    if size_b < 12:
        return 8192    # 7b / 8b
    if size_b <= 25:
        return 32768   # 14b / 20b
    return 131072      # 70b+


def build_graph(
    database_service: BaseDbRepository,
    embedding_service: TextEmbeddingService,
    model_name: str = 'llama3.1:8b',
):
    context_window = _context_window_from_model(model_name)
    logger.info(f"Model '{model_name}' → context_window={context_window}")

    llm = ChatOllama(
        model=model_name,
        temperature=0,
        num_ctx=context_window,
        keep_alive='5m',
        repeat_penalty=1.0
    )

    available_sources = database_service.get_all_filenames()
    router_nodes = GeneralNodes(llm, available_sources=available_sources)
    rag_nodes = RagNodes(llm, database_service, embedding_service, context_window=context_window)

    builder = StateGraph(AgentState)
    # general
    builder.add_node(NodeName.ROUTER, router_nodes.router_agent)
    builder.add_node(NodeName.GENERAL, router_nodes.general_agent)
    # rag
    builder.add_node(NodeName.QUERY_DECOMPOSER, rag_nodes.query_decomposer_agent)
    builder.add_node(NodeName.RESEARCH_WORKER, rag_nodes.research_worker)
    builder.add_node(NodeName.RETRIEVAL_GRADER_AGENT, rag_nodes.retrieval_grader_agent)
    builder.add_node(NodeName.FACT_EXTRACTOR, rag_nodes.fact_extractor_agent)
    builder.add_node(NodeName.SYNTHESIZER, rag_nodes.synthesizer_agent)
    builder.add_node(NodeName.COMPLETENESS_CHECKER, rag_nodes.completeness_checker_agent)
    builder.add_node(NodeName.HALLUCINATION_GRADER_AGENT, rag_nodes.hallucination_grader_agent)
    builder.add_node(NodeName.ERROR, rag_nodes.error_agent)

    builder.add_edge(START, NodeName.ROUTER)
    # general
    builder.add_conditional_edges(
        NodeName.ROUTER,
        GeneralNodes.route_intent,
        path_map={
            NodeName.QUERY_DECOMPOSER: NodeName.QUERY_DECOMPOSER,
            NodeName.GENERAL: NodeName.GENERAL,
        }
    )
    builder.add_edge(NodeName.GENERAL, END)
    # rag
    builder.add_conditional_edges(
        NodeName.QUERY_DECOMPOSER,
        RagNodes.validate_and_map,
        path_map={NodeName.RESEARCH_WORKER: NodeName.RESEARCH_WORKER, NodeName.ERROR: NodeName.ERROR}
    )
    builder.add_edge(NodeName.ERROR, END)

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
            END: END
        }
    )

    return builder.compile()
