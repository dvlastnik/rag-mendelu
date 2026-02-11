from enum import Enum

class NodeName(str, Enum):
    ROUTER = 'router_agent'
    GENERAL = 'general_agent'
    QUERY_REWRITER = 'query_rewriter_agent'
    QUERY_VERIFIER = 'query_verifier_agent'
    EXTRACTOR = 'extractor_agent'
    RESEARCH_WORKER = 'research_worker'
    RETRIEVAL_GRADER_AGENT = 'retrieval_grader_agent'
    HALLUCINATION_GRADER_AGENT = 'hallucination_grader_agent'
    SYNTHESIZER = 'synthesizer_agent'
    ERROR = 'error_agent'

class Intent(str, Enum):
    GENERAL = 'general'
    RAG = 'rag'