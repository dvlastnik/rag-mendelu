import pytest
from unittest.mock import MagicMock, ANY
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.types import Send
from langgraph.graph import END

from rag.agents.nodes.rag_nodes import RagNodes
from rag.agents.enums import NodeName
from rag.agents.models import GradeDocuments, GradeHallucinations

# ==========================================
# 1. QUERY REWRITER TESTS
# ==========================================
def test_query_rewriter_positive(rag_nodes, mock_llm):
    """Positive: LLM rewrites the query successfully."""
    mock_llm.invoke.return_value = AIMessage(content="Rewritten Query")
    
    state = {'messages': [HumanMessage(content="original")]}
    result = rag_nodes.query_rewriter_agent(state)
    
    assert result['rewritten_query'] == "Rewritten Query"

    args, _ = mock_llm.invoke.call_args
    assert "You are a Query Reformulator" in args[0][0].content

# ==========================================
# 3. RESEARCH WORKER TESTS
# ==========================================

def test_research_worker_negative_embedding_fail(rag_nodes, mock_embedding):
    """Negative: Embedding service returns None or error."""
    mock_embedding.get_embedding_with_uuid.return_value = []
    
    target = MagicMock()
    target.topics = []
    state = {'target': target, 'query': 'search text'}
    
    result = rag_nodes.research_worker(state)
    
    assert "Could not generate embeddings" in result['search_results'][0]

# ==========================================
# 4. RETRIEVAL GRADER TESTS
# ==========================================

def test_retrieval_grader_filtering(rag_nodes, mock_llm):
    """Positive: Filters out irrelevant docs using Batch/Listwise grading."""
    mock_batch_response = MagicMock()
    mock_batch_response.relevant_indices = [0] 
    
    mock_runnable = MagicMock()
    mock_runnable.invoke.return_value = mock_batch_response
    mock_llm.with_structured_output.return_value = mock_runnable
    
    state = {
        'rewritten_query': 'q', 
        'search_results': ['Good Doc', 'Bad Doc']
    }
    
    result = rag_nodes.retrieval_grader_agent(state)
    
    assert len(result['filtered_results']) == 1
    assert result['filtered_results'][0] == "Good Doc"
    
    mock_runnable.invoke.assert_called_once()

def test_retrieval_grader_empty_input(rag_nodes):
    """Negative: Handles empty search results gracefully."""
    state = {'rewritten_query': 'q', 'search_results': []}
    result = rag_nodes.retrieval_grader_agent(state)
    assert result['filtered_results'] == []

# ==========================================
# 5. SYNTHESIZER TESTS
# ==========================================

def test_synthesizer_positive_clean(rag_nodes, mock_llm):
    """Positive: Generates answer from context."""
    mock_llm.invoke.return_value = AIMessage(content="Final Answer")
    
    state = {
        'messages': [HumanMessage(content="User Q")],
        'filtered_results': ["Doc 1", "Doc 2"],
        'hallucination_status': 'clean'
    }
    
    result = rag_nodes.synthesizer_agent(state)
    assert result['messages'][0].content == "Final Answer"
    
    args, _ = mock_llm.invoke.call_args
    user_prompt = args[0][1].content
    assert "CRITICAL WARNING" not in user_prompt

def test_synthesizer_retry_warning_injection(rag_nodes, mock_llm):
    """Positive: Injects warning if previous attempt was hallucinated."""
    mock_llm.invoke.return_value = AIMessage(content="Fixed Answer")
    
    state = {
        'messages': [HumanMessage(content="User Q")],
        'filtered_results': ["Doc 1"],
        'hallucination_status': 'hallucinated'
    }
    
    rag_nodes.synthesizer_agent(state)
    
    args, _ = mock_llm.invoke.call_args
    user_prompt = args[0][1].content
    assert "CRITICAL WARNING" in user_prompt

def test_synthesizer_negative_no_docs(rag_nodes):
    """Negative: No documents found -> I cannot answer."""
    state = {
        'messages': [HumanMessage(content="User Q")],
        'filtered_results': []
    }
    result = rag_nodes.synthesizer_agent(state)
    assert "could not find specific information" in result['messages'][0].content

# ==========================================
# 6. HALLUCINATION GRADER TESTS
# ==========================================
def test_hallucination_grader_positive_grounded(rag_nodes, mock_llm):
    """Positive: Document is grounded."""
    mock_grade = MagicMock()
    mock_grade.is_relevant = "yes"
    
    mock_runnable = MagicMock()
    mock_runnable.invoke.return_value = mock_grade
    mock_llm.with_structured_output.return_value = mock_runnable
    
    state = {
        'filtered_results': ['doc'], 
        'messages': [AIMessage(content='answer')]
    }
    
    result = rag_nodes.hallucination_grader_agent(state)
    assert result['hallucination_status'] == 'clean'

def test_hallucination_grader_negative_hallucinated(rag_nodes, mock_llm):
    """Negative: Hallucination detected, increments retries."""
    mock_grade = MagicMock()
    mock_grade.is_relevant = "no"
    
    mock_runnable = MagicMock()
    mock_runnable.invoke.return_value = mock_grade
    mock_llm.with_structured_output.return_value = mock_runnable
    
    state = {
        'filtered_results': ['doc'], 
        'messages': [AIMessage(content='answer')],
        'hallucination_retries': 1
    }
    
    result = rag_nodes.hallucination_grader_agent(state)
    assert result['hallucination_status'] == 'hallucinated'
    assert result['hallucination_retries'] == 2

# ==========================================
# 7. ROUTING LOGIC (Static Methods)
# ==========================================
def test_validate_and_map_fan_out():
    """Positive: Maps 2 extracted targets to 2 Workers."""
    t1 = MagicMock(city="A")
    t2 = MagicMock(city="B")
    state = {
        'extracted_data': [t1, t2],
        'rewritten_query': 'q'
    }
    
    result = RagNodes.validate_and_map(state)
    
    assert len(result) == 2
    assert isinstance(result[0], Send)
    assert result[0].node == NodeName.RESEARCH_WORKER

def test_route_hallucination_retry():
    """Positive: Loop back to Synthesizer."""
    state = {'hallucination_status': 'hallucinated', 'hallucination_retries': 1}
    assert RagNodes.route_hallucination(state) == NodeName.SYNTHESIZER

def test_route_hallucination_stop():
    """Negative: Max retries reached -> END."""
    state = {'hallucination_status': 'hallucinated', 'hallucination_retries': 3}
    assert RagNodes.route_hallucination(state) == END