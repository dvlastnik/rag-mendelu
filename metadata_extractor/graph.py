from typing import Optional, Dict, Any
from langgraph.graph import StateGraph, END, START
from langchain_ollama import ChatOllama
from langchain_core.language_models import BaseChatModel

from metadata_extractor.enums import NodeName
from metadata_extractor.state import ExtractorAgentState
from metadata_extractor.nodes import Node


def build_extractor_graph(
    model_name: str = 'llama3.1:8b',
    llm: Optional[BaseChatModel] = None,
    temperature: float = 0.0,
    num_ctx: int = 4096,
    timeout: float = 120.0,
) -> StateGraph:
    """
    Build the metadata extraction graph.
    
    Args:
        model_name: Ollama model name (ignored if llm is provided).
        llm: Optional pre-configured LLM instance.
        temperature: LLM temperature (0 = deterministic).
        num_ctx: Context window size.
        timeout: Request timeout in seconds.
        
    Returns:
        Compiled LangGraph ready for invocation.
        
    Example:
        >>> graph = build_extractor_graph()
        >>> result = graph.invoke({'text_chunk': 'Floods in Pakistan, 2022'})
        >>> result['clean_data']
        {'years': [2022], 'locations': ['Pakistan']}
    """
    graph = StateGraph(ExtractorAgentState)

    if llm is None:
        llm = ChatOllama(
            model=model_name,
            temperature=temperature,
            num_ctx=num_ctx,
            keep_alive='5m',
            num_predict=512,
            timeout=timeout,
        )
    
    nodes = Node(llm)

    # nodes
    graph.add_node(NodeName.EXTRACTOR, nodes.extraction_agent)
    graph.add_node(NodeName.CLEAN, nodes.cleaning_agent)

    # edges
    graph.add_edge(START, NodeName.EXTRACTOR)
    graph.add_edge(NodeName.EXTRACTOR, NodeName.CLEAN)
    graph.add_edge(NodeName.CLEAN, END)

    return graph.compile()


def extract_metadata(text: str, graph=None, **kwargs) -> Dict[str, Any]:
    """
    Convenience function for one-off metadata extraction.
    
    Args:
        text: Text to extract metadata from.
        graph: Optional pre-built graph (builds new one if None).
        **kwargs: Passed to build_extractor_graph if building new graph.
        
    Returns:
        Dict with 'years' (List[int]) and 'locations' (List[str]).
    """
    if graph is None:
        graph = build_extractor_graph(**kwargs)
    
    result = graph.invoke({'text_chunk': text})
    return result.get('clean_data', {'years': [], 'locations': []})