import pytest
from unittest.mock import MagicMock, call
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from rag.agents.nodes.scientific_nodes import ScientificNodes
from rag.agents.scientific_source import ScientificSearchResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(source_id: str, variable: str, text: str = "some data", score: float = 0.9):
    return ScientificSearchResult(text=text, score=score, source_id=source_id, variable=variable)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_scientific_source():
    src = MagicMock()
    src.search.return_value = []
    return src


@pytest.fixture
def scientific_nodes(mock_llm, mock_scientific_source):
    return ScientificNodes(scientific_source=mock_scientific_source, llm=mock_llm)


# ==========================================
# 1. SCIENTIFIC RETRIEVER TESTS
# ==========================================

def test_scientific_retriever_uses_rewritten_queries(scientific_nodes, mock_scientific_source):
    """Positive: rewritten_queries are used as search queries."""
    mock_scientific_source.search.return_value = [_make_result("era5", "temp")]

    state = {
        'rewritten_queries': ["temperature anomaly 2020"],
        'messages': [HumanMessage(content="original question")],
    }
    result = scientific_nodes.scientific_retriever(state)

    mock_scientific_source.search.assert_called_once_with("temperature anomaly 2020", top_k=8)
    assert len(result['scientific_results']) == 1


def test_scientific_retriever_falls_back_to_human_message(scientific_nodes, mock_scientific_source):
    """When rewritten_queries is empty, last HumanMessage content is used."""
    mock_scientific_source.search.return_value = [_make_result("era5", "precip")]

    state = {
        'rewritten_queries': [],
        'messages': [HumanMessage(content="what is the precipitation data?")],
    }
    result = scientific_nodes.scientific_retriever(state)

    mock_scientific_source.search.assert_called_once_with("what is the precipitation data?", top_k=8)
    assert len(result['scientific_results']) == 1


def test_scientific_retriever_caps_at_three_queries(scientific_nodes, mock_scientific_source):
    """Only the first 3 of rewritten_queries are searched."""
    mock_scientific_source.search.return_value = []

    state = {
        'rewritten_queries': ["q1", "q2", "q3", "q4", "q5"],
        'messages': [],
    }
    scientific_nodes.scientific_retriever(state)

    assert mock_scientific_source.search.call_count == 3
    calls = [c.args[0] for c in mock_scientific_source.search.call_args_list]
    assert calls == ["q1", "q2", "q3"]


def test_scientific_retriever_deduplicates_by_source_variable(scientific_nodes, mock_scientific_source):
    """Two results with the same source_id:variable key → only first is kept."""
    dup_a = _make_result("era5", "temp", text="first hit", score=0.95)
    dup_b = _make_result("era5", "temp", text="second hit", score=0.80)
    unique = _make_result("era5", "precip", text="unique", score=0.70)
    mock_scientific_source.search.return_value = [dup_a, dup_b, unique]

    state = {
        'rewritten_queries': ["climate query"],
        'messages': [],
    }
    result = scientific_nodes.scientific_retriever(state)

    texts = [r.text for r in result['scientific_results']]
    assert "first hit" in texts
    assert "second hit" not in texts
    assert "unique" in texts
    assert len(result['scientific_results']) == 2


def test_scientific_retriever_returns_empty_on_no_results(scientific_nodes, mock_scientific_source):
    """search returns [] → result list is empty."""
    mock_scientific_source.search.return_value = []

    state = {
        'rewritten_queries': ["some query"],
        'messages': [],
    }
    result = scientific_nodes.scientific_retriever(state)

    assert result == {'scientific_results': []}


def test_scientific_retriever_handles_search_exception(scientific_nodes, mock_scientific_source):
    """search raises Exception → no crash, returns empty results."""
    mock_scientific_source.search.side_effect = Exception("connection refused")

    state = {
        'rewritten_queries': ["failing query"],
        'messages': [],
    }
    result = scientific_nodes.scientific_retriever(state)

    assert result == {'scientific_results': []}


# ==========================================
# 2. MULTI SOURCE SYNTHESIZER TESTS
# ==========================================

def test_multi_source_synthesizer_returns_ai_message(scientific_nodes, mock_llm):
    """Positive: returns {"messages": [AIMessage]}."""
    mock_llm.invoke.return_value = AIMessage(content="combined answer")

    state = {
        'scientific_results': [_make_result("era5", "temp")],
        'distilled_facts': ["Drought increased in 2020."],
    }
    result = scientific_nodes.multi_source_synthesizer(state)

    assert "messages" in result
    assert len(result['messages']) == 1
    assert isinstance(result['messages'][0], AIMessage)


def test_multi_source_synthesizer_calls_llm_with_system_prompt(scientific_nodes, mock_llm):
    """LLM is invoked with a SystemMessage as the first message."""
    mock_llm.invoke.return_value = AIMessage(content="answer")

    state = {'scientific_results': [], 'distilled_facts': []}
    scientific_nodes.multi_source_synthesizer(state)

    args, _ = mock_llm.invoke.call_args
    messages = args[0]
    assert isinstance(messages[0], SystemMessage)
    assert "Climate Datasets" in messages[0].content or "synthesis" in messages[0].content.lower()


def test_multi_source_synthesizer_includes_scientific_results_in_prompt(scientific_nodes, mock_llm):
    """source_id, variable, and text from scientific results appear in HumanMessage."""
    mock_llm.invoke.return_value = AIMessage(content="answer")

    result_obj = _make_result("era5_hourly", "temperature_2m", text="mean temp 14.5°C", score=0.88)
    state = {
        'scientific_results': [result_obj],
        'distilled_facts': [],
    }
    scientific_nodes.multi_source_synthesizer(state)

    args, _ = mock_llm.invoke.call_args
    human_content = args[0][1].content
    assert "era5_hourly" in human_content
    assert "temperature_2m" in human_content
    assert "mean temp 14.5°C" in human_content


def test_multi_source_synthesizer_includes_distilled_facts_in_prompt(scientific_nodes, mock_llm):
    """distilled_facts strings appear in the HumanMessage content."""
    mock_llm.invoke.return_value = AIMessage(content="answer")

    state = {
        'scientific_results': [],
        'distilled_facts': ["Precipitation dropped 30% in 2019.", "Heatwaves were more frequent."],
    }
    scientific_nodes.multi_source_synthesizer(state)

    args, _ = mock_llm.invoke.call_args
    human_content = args[0][1].content
    assert "Precipitation dropped 30%" in human_content
    assert "Heatwaves were more frequent" in human_content


def test_multi_source_synthesizer_handles_empty_scientific_results(scientific_nodes, mock_llm):
    """Empty scientific_results → fallback text in HumanMessage."""
    mock_llm.invoke.return_value = AIMessage(content="answer")

    state = {'scientific_results': [], 'distilled_facts': ["Some fact."]}
    scientific_nodes.multi_source_synthesizer(state)

    args, _ = mock_llm.invoke.call_args
    human_content = args[0][1].content
    assert "No scientific dataset results available" in human_content


def test_multi_source_synthesizer_handles_empty_distilled_facts(scientific_nodes, mock_llm):
    """Empty distilled_facts → fallback text in HumanMessage."""
    mock_llm.invoke.return_value = AIMessage(content="answer")

    state = {
        'scientific_results': [_make_result("era5", "wind")],
        'distilled_facts': [],
    }
    scientific_nodes.multi_source_synthesizer(state)

    args, _ = mock_llm.invoke.call_args
    human_content = args[0][1].content
    assert "No document facts available" in human_content
