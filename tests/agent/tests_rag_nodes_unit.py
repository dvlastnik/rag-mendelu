import pytest
from unittest.mock import MagicMock, ANY
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.types import Send
from langgraph.graph import END

from rag.agents.nodes.rag_nodes import RagNodes
from rag.agents.enums import NodeName, Intent
from rag.agents.models import GradeDocuments, GradeHallucinations, QueryPlan, QueryStrategy

# ==========================================
# 1. QUERY PLANNER TESTS
# ==========================================
def test_query_planner_fallback_to_vector(rag_nodes, mock_llm):
    """Positive: Without DuckDB catalog, falls back to vector decomposition."""
    # rag_nodes fixture has no duck_db_repo, so _compact_catalog is ""
    mock_runnable = MagicMock()
    mock_runnable.invoke.return_value = MagicMock(queries=["keyword query", "conceptual query"])
    mock_llm.with_structured_output.return_value = mock_runnable

    state = {
        'messages': [HumanMessage(content="original question")],
        'detected_source': None,
        'intent': Intent.RAG,
    }
    result = rag_nodes.query_planner_agent(state)

    assert 'rewritten_queries' in result
    assert "original question" in result['rewritten_queries']
    assert result['query_plan'].strategy == QueryStrategy.VECTOR

# ==========================================
# 2. RESEARCH WORKER TESTS
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
# 3. RETRIEVAL GRADER TESTS
# ==========================================

def test_retrieval_grader_reranking(rag_nodes):
    """Positive: Reranks docs and returns top-N."""
    doc1 = MagicMock(id="1", text="Good Doc", metadata={"source": "test"})
    doc2 = MagicMock(id="2", text="Bad Doc", metadata={"source": "test"})

    state = {
        'messages': [HumanMessage(content='q')],
        'search_results': [doc1, doc2],
        'filtered_results': [],
        'intent': Intent.RAG,
    }

    result = rag_nodes.retrieval_grader_agent(state)

    assert len(result['filtered_results']) <= rag_nodes.distiller_top_n

def test_retrieval_grader_empty_input(rag_nodes):
    """Negative: Handles empty search results gracefully."""
    state = {
        'messages': [HumanMessage(content='q')],
        'search_results': [],
        'filtered_results': [],
        'intent': Intent.RAG,
    }
    result = rag_nodes.retrieval_grader_agent(state)
    assert result['filtered_results'] == []

def test_retrieval_grader_excludes_already_filtered(rag_nodes):
    """Reranker skips docs that are already in filtered_results from prior iterations."""
    existing_doc = MagicMock(id="1", text="Already filtered", metadata={"source": "test"})
    new_doc = MagicMock(id="2", text="New doc", metadata={"source": "test"})

    state = {
        'messages': [HumanMessage(content='q')],
        'search_results': [existing_doc, new_doc],
        'filtered_results': [existing_doc],
        'intent': Intent.RAG,
    }

    result = rag_nodes.retrieval_grader_agent(state)

    result_ids = [doc.id for doc in result['filtered_results']]
    assert "1" not in result_ids

def test_retrieval_grader_adaptive_topn_exhaustive(rag_nodes):
    """Exhaustive intent uses 3x top_n."""
    docs = [MagicMock(id=str(i), text=f"Doc {i}", metadata={"source": "test"}) for i in range(30)]

    state = {
        'messages': [HumanMessage(content='list all bands')],
        'search_results': docs,
        'filtered_results': [],
        'intent': Intent.RAG_EXHAUSTIVE,
    }

    result = rag_nodes.retrieval_grader_agent(state)

    assert len(result['filtered_results']) <= rag_nodes.distiller_top_n * 3

# ==========================================
# 4. SCROLL RETRIEVER TESTS
# ==========================================
def test_scroll_retriever_fetches_docs(rag_nodes, mock_db):
    """Scroll retriever fetches all docs from a source."""
    mock_docs = [MagicMock(id=str(i)) for i in range(5)]
    mock_db.scroll_all_by_source.return_value = mock_docs

    state = {'detected_source': 'history_of_metal'}
    result = rag_nodes.scroll_retriever(state)

    assert len(result['filtered_results']) == 5
    assert len(result['search_results']) == 5
    mock_db.scroll_all_by_source.assert_called_once_with('history_of_metal', limit=500)

def test_scroll_retriever_no_source(rag_nodes):
    """Scroll retriever returns empty when no source detected."""
    state = {'detected_source': None}
    result = rag_nodes.scroll_retriever(state)

    assert result['filtered_results'] == []

# ==========================================
# 5. SYNTHESIZER TESTS
# ==========================================

def test_synthesizer_positive_clean(rag_nodes, mock_llm):
    """Positive: Generates answer from distilled facts."""
    mock_llm.invoke.return_value = AIMessage(content="Final Answer")

    state = {
        'messages': [HumanMessage(content="User Q")],
        'distilled_facts': ["[Sources: test]\nFact 1\nFact 2"],
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
        'distilled_facts': ["[Sources: test]\nFact 1"],
        'hallucination_status': 'hallucinated'
    }

    rag_nodes.synthesizer_agent(state)

    args, _ = mock_llm.invoke.call_args
    user_prompt = args[0][1].content
    assert "CRITICAL WARNING" in user_prompt

def test_synthesizer_negative_no_facts(rag_nodes):
    """Negative: No distilled facts -> I cannot answer."""
    state = {
        'messages': [HumanMessage(content="User Q")],
        'distilled_facts': []
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
        'filtered_results': [MagicMock(text='doc')],
        'distilled_facts': ["[Sources: test]\nSome fact"],
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
        'filtered_results': [MagicMock(text='doc')],
        'distilled_facts': ["[Sources: test]\nSome fact"],
        'messages': [AIMessage(content='answer')],
        'hallucination_retries': 1
    }

    result = rag_nodes.hallucination_grader_agent(state)
    assert result['hallucination_status'] == 'hallucinated'
    assert result['hallucination_retries'] == 2

# ==========================================
# 7. ROUTING LOGIC (Static Methods)
# ==========================================
def test_route_query_plan_vector_fan_out():
    """Positive: VECTOR strategy fans out to RESEARCH_WORKER per query."""
    state = {
        'extracted_data': [],
        'rewritten_queries': ['q1', 'q2'],
        'messages': [HumanMessage(content='q')],
        'query_plan': QueryPlan(strategy=QueryStrategy.VECTOR, vector_queries=['q1', 'q2']),
    }

    result = RagNodes.route_query_plan(state)

    assert len(result) == 2
    assert isinstance(result[0], Send)
    assert result[0].node == NodeName.RESEARCH_WORKER

def test_route_query_plan_sql_routes_to_analytical():
    """SQL strategy routes to ANALYTICAL_QUERY node."""
    state = {
        'rewritten_queries': [],
        'messages': [HumanMessage(content='what is the highest rated game?')],
        'query_plan': QueryPlan(strategy=QueryStrategy.SQL, sql_sources=['games_2025'], sql_hint='max review'),
    }

    result = RagNodes.route_query_plan(state)
    assert result == NodeName.ANALYTICAL_QUERY

def test_route_query_plan_scroll_routes_to_scroll_retriever():
    """SCROLL strategy routes to SCROLL_RETRIEVER."""
    state = {
        'rewritten_queries': [],
        'messages': [HumanMessage(content='summarize history_of_metal')],
        'query_plan': QueryPlan(strategy=QueryStrategy.SCROLL),
    }

    result = RagNodes.route_query_plan(state)
    assert result == NodeName.SCROLL_RETRIEVER

def test_route_query_plan_no_plan_falls_back_to_vector():
    """No query_plan in state falls back to vector fan-out."""
    state = {
        'extracted_data': [],
        'rewritten_queries': ['q1'],
        'messages': [HumanMessage(content='question')],
        'query_plan': None,
    }

    result = RagNodes.route_query_plan(state)
    assert isinstance(result, list)
    assert isinstance(result[0], Send)

def test_route_hallucination_retry():
    """Positive: Loop back to Synthesizer."""
    state = {'hallucination_status': 'hallucinated', 'hallucination_retries': 1}
    assert RagNodes.route_hallucination(state) == NodeName.SYNTHESIZER

def test_route_hallucination_stop():
    """Negative: Max retries reached -> END."""
    state = {
        'hallucination_status': 'hallucinated',
        'hallucination_retries': 3,
        'messages': [AIMessage(content='some answer')],
    }
    assert RagNodes.route_hallucination(state) == END

# ==========================================
# 8. ROUTE COMPLETENESS CHECK
# ==========================================
def test_route_completeness_skip_for_summarization_scroll(rag_nodes):
    """Summarization scroll-based queries skip gap-check loop."""
    state = {
        'retrieval_iterations': 0,
        'completeness_follow_up_query': 'some follow up',
        'intent': Intent.RAG_SUMMARIZATION,
        'detected_source': 'history_of_metal',
        'search_results': [],
    }

    result = rag_nodes.route_completeness_check(state)
    assert result == NodeName.HALLUCINATION_GRADER_AGENT

def test_route_completeness_normal_follow_up(rag_nodes):
    """Normal RAG queries with follow-up trigger another iteration."""
    state = {
        'retrieval_iterations': 1,
        'completeness_follow_up_query': 'missing info keywords',
        'intent': Intent.RAG,
        'detected_source': None,
        'search_results': [],
    }

    result = rag_nodes.route_completeness_check(state)
    assert isinstance(result, list)
    assert isinstance(result[0], Send)

def test_route_completeness_exhaustive_uses_follow_up(rag_nodes):
    """Exhaustive + incomplete routes to RESEARCH_WORKER (same as RAG)."""
    state = {
        'retrieval_iterations': 1,
        'completeness_follow_up_query': 'more bands missing',
        'intent': Intent.RAG_EXHAUSTIVE,
        'detected_source': None,
        'search_results': [],
    }

    result = rag_nodes.route_completeness_check(state)
    assert isinstance(result, list)
    assert isinstance(result[0], Send)
    assert result[0].node == NodeName.RESEARCH_WORKER
