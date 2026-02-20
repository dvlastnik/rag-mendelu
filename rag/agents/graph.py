from langgraph.graph import StateGraph, START, END
#from langchain.chat_models import init_chat_model
from langchain_ollama import ChatOllama

from rag.agents.state import AgentState
from rag.agents.enums import NodeName
from rag.agents.nodes.general_nodes import GeneralNodes
from rag.agents.nodes.rag_nodes import RagNodes

from database.base.BaseDbRepository import BaseDbRepository
from text_embedding import TextEmbeddingService
from utils.logging_config import get_logger

logger = get_logger(__name__)

def build_graph(database_service: BaseDbRepository, embedding_service: TextEmbeddingService, model_name: str = 'ibm/granite4:3b'):
    llm = ChatOllama(
        model=model_name,
        temperature=0,
        num_ctx=8192, 
        keep_alive='5m'
    )
    
    router_nodes = GeneralNodes(llm)
    rag_nodes = RagNodes(llm, database_service, embedding_service)

    builder = StateGraph(AgentState)
    # general
    builder.add_node(NodeName.ROUTER, router_nodes.router_agent)
    builder.add_node(NodeName.GENERAL, router_nodes.general_agent)
    # rag
    builder.add_node(NodeName.QUERY_REWRITER, rag_nodes.query_rewriter_agent)
    builder.add_node(NodeName.RESEARCH_WORKER, rag_nodes.research_worker)
    builder.add_node(NodeName.RETRIEVAL_GRADER_AGENT, rag_nodes.retrieval_grader_agent)
    builder.add_node(NodeName.CONTEXT_COMPRESSOR, rag_nodes.context_compressor_agent)
    builder.add_node(NodeName.HALLUCINATION_GRADER_AGENT, rag_nodes.hallucination_grader_agent)
    builder.add_node(NodeName.SYNTHESIZER, rag_nodes.synthesizer_agent)
    builder.add_node(NodeName.ERROR, rag_nodes.error_agent)

    builder.add_edge(START, NodeName.ROUTER)
    # general
    builder.add_conditional_edges(
        NodeName.ROUTER,
        GeneralNodes.route_intent, 
        path_map={NodeName.QUERY_REWRITER: NodeName.QUERY_REWRITER, NodeName.GENERAL: NodeName.GENERAL}
    )
    builder.add_edge(NodeName.GENERAL, END)
    # rag
    builder.add_conditional_edges(
        NodeName.QUERY_REWRITER,
        RagNodes.validate_and_map,
        path_map={NodeName.RESEARCH_WORKER: NodeName.RESEARCH_WORKER, NodeName.ERROR: NodeName.ERROR}
    )
    builder.add_edge(NodeName.ERROR, END)

    builder.add_edge(NodeName.RESEARCH_WORKER, NodeName.RETRIEVAL_GRADER_AGENT)
    builder.add_edge(NodeName.RETRIEVAL_GRADER_AGENT, NodeName.CONTEXT_COMPRESSOR)
    builder.add_edge(NodeName.CONTEXT_COMPRESSOR, NodeName.SYNTHESIZER)
    builder.add_edge(NodeName.SYNTHESIZER, NodeName.HALLUCINATION_GRADER_AGENT)

    builder.add_conditional_edges(
        NodeName.HALLUCINATION_GRADER_AGENT,
        rag_nodes.route_hallucination,
        path_map={
            NodeName.SYNTHESIZER: NodeName.SYNTHESIZER,
            END: END
        }
    )
    
    return builder.compile()