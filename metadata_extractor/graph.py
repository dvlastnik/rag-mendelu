from langgraph.graph import StateGraph, END, START
from langchain_ollama import ChatOllama

from metadata_extractor.enums import NodeName
from metadata_extractor.state import ExtractorAgentState
from metadata_extractor.nodes import Node

def build_extractor_graph(model_name: str = 'ibm/granite4:3b'):
    graph = StateGraph(ExtractorAgentState)

    llm = ChatOllama(
        model=model_name,
        temperature=0,
        num_ctx=4096, 
        keep_alive='5m',
        num_predict=1024,
        timeout=120.0
    )
    nodes = Node(llm)

    graph.add_node(NodeName.EXTRACTOR, nodes.extraction_agent)
    graph.add_node(NodeName.NORMALIZATION, nodes.normalization_agent)
    graph.add_node(NodeName.CLEAN, nodes.cleaning_agent)

    graph.add_edge(START, NodeName.EXTRACTOR)
    graph.add_edge(NodeName.EXTRACTOR, NodeName.NORMALIZATION)
    graph.add_edge(NodeName.NORMALIZATION, NodeName.CLEAN)
    graph.add_edge(NodeName.CLEAN, END)

    return graph.compile()