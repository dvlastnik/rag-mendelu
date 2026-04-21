import pytest
from unittest.mock import MagicMock
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from rag.agents.nodes.general_nodes import GeneralNodes
from rag.agents.models import GeneralOrRagDecision
from rag.agents.enums import NodeName, Intent

# ==========================================
# 1. ROUTER AGENT TESTS
# ==========================================
def test_router_agent_identifies_rag_intent(mock_llm):
    """
    Positive Test: Ensure complex queries are routed to RAG.
    """
    mock_runnable = MagicMock()
    mock_runnable.invoke.return_value = GeneralOrRagDecision(intent=Intent.RAG)
    mock_llm.with_structured_output.return_value = mock_runnable

    state = {'messages': [HumanMessage(content="What is the weather in Prague?")]}
    node = GeneralNodes(mock_llm)

    result = node.router_agent(state)

    assert result['intent'] == Intent.RAG
    mock_runnable.invoke.assert_called_once()


def test_router_agent_identifies_general_intent(mock_llm):
    """
    Negative/Alternative Test: Ensure greetings are routed to GENERAL (not RAG).
    """
    mock_runnable = MagicMock()
    mock_runnable.invoke.return_value = GeneralOrRagDecision(intent=Intent.GENERAL)
    mock_llm.with_structured_output.return_value = mock_runnable

    state = {'messages': [HumanMessage(content="Hi there!")]}
    node = GeneralNodes(mock_llm)

    result = node.router_agent(state)

    assert result['intent'] == Intent.GENERAL


def test_router_agent_keyword_upgrade_to_exhaustive(mock_llm):
    """Keyword fallback upgrades RAG to RAG_EXHAUSTIVE for listing queries."""
    mock_runnable = MagicMock()
    mock_runnable.invoke.return_value = GeneralOrRagDecision(intent=Intent.RAG)
    mock_llm.with_structured_output.return_value = mock_runnable

    state = {'messages': [HumanMessage(content="List all music bands mentioned in the documents")]}
    node = GeneralNodes(mock_llm)

    result = node.router_agent(state)

    assert result['intent'] == Intent.RAG_EXHAUSTIVE


def test_router_agent_keyword_upgrade_to_summarization(mock_llm):
    """Keyword fallback upgrades RAG to RAG_SUMMARIZATION for summarization queries."""
    mock_runnable = MagicMock()
    mock_runnable.invoke.return_value = GeneralOrRagDecision(intent=Intent.RAG)
    mock_llm.with_structured_output.return_value = mock_runnable

    state = {'messages': [HumanMessage(content="Summarize the drought report")]}
    node = GeneralNodes(mock_llm)

    result = node.router_agent(state)

    assert result['intent'] == Intent.RAG_SUMMARIZATION


def test_router_agent_source_matching(mock_llm):
    """Fuzzy-matches detected_source against available_sources."""
    mock_runnable = MagicMock()
    mock_runnable.invoke.return_value = GeneralOrRagDecision(
        intent=Intent.RAG_SUMMARIZATION,
        detected_source="history of metal"
    )
    mock_llm.with_structured_output.return_value = mock_runnable

    state = {'messages': [HumanMessage(content="Summarize history_of_metal")]}
    node = GeneralNodes(mock_llm, available_sources=["history_of_metal", "games_2025", "drough"])

    result = node.router_agent(state)

    assert result['detected_source'] == 'history_of_metal'


def test_router_agent_handles_string_input_failure(mock_llm):
    """
    Negative Test: Ensure we catch format errors if someone passes a string
    instead of a Message object (reproducing your earlier bug).
    """
    node = GeneralNodes(mock_llm)

    bad_state = {'messages': ["this is a string, not a Message object"]}

    with pytest.raises(AttributeError):
        node.router_agent(bad_state)


# ==========================================
# 2. GENERAL AGENT TESTS
# ==========================================
def test_general_agent_generates_response(mock_llm):
    """
    Positive Test: Ensure the general agent calls the LLM and returns a message list.
    """
    expected_response_text = "Hello! How can I help you?"
    mock_llm.invoke.return_value = AIMessage(content=expected_response_text)

    state = {'messages': [HumanMessage(content="Hello")]}
    node = GeneralNodes(mock_llm)

    result = node.general_agent(state)

    assert "messages" in result
    assert isinstance(result["messages"], list)

    generated_msg = result["messages"][0]
    assert isinstance(generated_msg, AIMessage)
    assert generated_msg.content == expected_response_text

    args, _ = mock_llm.invoke.call_args
    assert isinstance(args[0][0], SystemMessage)
    assert "You are a helpful assistant" in args[0][0].content


# ==========================================
# 3. STATIC METHOD TESTS (Logic only)
# ==========================================
def test_route_intent_logic_rag():
    """Positive Test: RAG intent goes to Query Decomposer"""
    state = {'intent': Intent.RAG}
    next_node = GeneralNodes.route_intent(state)
    assert next_node == NodeName.QUERY_PLANNER

def test_route_intent_logic_exhaustive():
    """Exhaustive intent also goes to Query Decomposer"""
    state = {'intent': Intent.RAG_EXHAUSTIVE}
    next_node = GeneralNodes.route_intent(state)
    assert next_node == NodeName.QUERY_PLANNER

def test_route_intent_logic_summarization():
    """Summarization intent also goes to Query Decomposer"""
    state = {'intent': Intent.RAG_SUMMARIZATION}
    next_node = GeneralNodes.route_intent(state)
    assert next_node == NodeName.QUERY_PLANNER

def test_route_intent_logic_general():
    """Negative Test: General intent goes to General Node"""
    state = {'intent': Intent.GENERAL}
    next_node = GeneralNodes.route_intent(state)
    assert next_node == NodeName.GENERAL


# ==========================================
# 4. KEYWORD UPGRADE LOGIC
# ==========================================
def test_keyword_upgrade_does_not_touch_general():
    """General intent should never be upgraded by keywords."""
    result = GeneralNodes._keyword_intent_upgrade("list all bands", Intent.GENERAL)
    assert result == Intent.GENERAL

def test_keyword_upgrade_list_all():
    result = GeneralNodes._keyword_intent_upgrade("list all European floods", Intent.RAG)
    assert result == Intent.RAG_EXHAUSTIVE

def test_keyword_upgrade_every():
    result = GeneralNodes._keyword_intent_upgrade("every band mentioned in the docs", Intent.RAG)
    assert result == Intent.RAG_EXHAUSTIVE

def test_keyword_upgrade_summarize():
    result = GeneralNodes._keyword_intent_upgrade("summarize the document about drought", Intent.RAG)
    assert result == Intent.RAG_SUMMARIZATION

def test_keyword_upgrade_no_match():
    result = GeneralNodes._keyword_intent_upgrade("What year did Metallica form?", Intent.RAG)
    assert result == Intent.RAG


# ==========================================
# 5. SCIENTIFIC INTENT UPGRADE & ROUTING
# ==========================================

# --- 5a. _scientific_intent_upgrade ---

def test_scientific_upgrade_triggers_on_keyword_with_source(mock_llm):
    """RAG intent + scientific keyword + source wired → SCIENTIFIC."""
    node = GeneralNodes(mock_llm, scientific_source=MagicMock())
    result = node._scientific_intent_upgrade("show me the NetCDF data", Intent.RAG)
    assert result == Intent.SCIENTIFIC


def test_scientific_upgrade_no_op_without_source(mock_llm):
    """Same query but scientific_source=None → intent unchanged."""
    node = GeneralNodes(mock_llm, scientific_source=None)
    result = node._scientific_intent_upgrade("show me the NetCDF data", Intent.RAG)
    assert result == Intent.RAG


def test_scientific_upgrade_does_not_touch_general(mock_llm):
    """GENERAL intent is never upgraded even with keywords and source wired."""
    node = GeneralNodes(mock_llm, scientific_source=MagicMock())
    result = node._scientific_intent_upgrade("NetCDF raster climate dataset", Intent.GENERAL)
    assert result == Intent.GENERAL


def test_scientific_upgrade_does_not_downgrade_multi_source(mock_llm):
    """MULTI_SOURCE stays MULTI_SOURCE even with scientific keywords."""
    node = GeneralNodes(mock_llm, scientific_source=MagicMock())
    result = node._scientific_intent_upgrade("compare raster data with documents", Intent.MULTI_SOURCE)
    assert result == Intent.MULTI_SOURCE


def test_scientific_upgrade_no_match_no_change(mock_llm):
    """No scientific keywords → intent unchanged."""
    node = GeneralNodes(mock_llm, scientific_source=MagicMock())
    result = node._scientific_intent_upgrade("What year did Metallica form?", Intent.RAG)
    assert result == Intent.RAG


# --- 5b. route_intent_with_scientific ---

def test_route_intent_scientific_goes_to_scientific_retriever(mock_llm):
    """SCIENTIFIC intent → SCIENTIFIC_RETRIEVER."""
    node = GeneralNodes(mock_llm, scientific_source=MagicMock())
    assert node.route_intent_with_scientific({'intent': Intent.SCIENTIFIC}) == NodeName.SCIENTIFIC_RETRIEVER


def test_route_intent_multi_source_goes_to_scientific_retriever(mock_llm):
    """MULTI_SOURCE intent → SCIENTIFIC_RETRIEVER (scientific runs first)."""
    node = GeneralNodes(mock_llm, scientific_source=MagicMock())
    assert node.route_intent_with_scientific({'intent': Intent.MULTI_SOURCE}) == NodeName.SCIENTIFIC_RETRIEVER


def test_route_intent_with_scientific_rag_goes_to_query_planner(mock_llm):
    """RAG intent → QUERY_PLANNER."""
    node = GeneralNodes(mock_llm, scientific_source=MagicMock())
    assert node.route_intent_with_scientific({'intent': Intent.RAG}) == NodeName.QUERY_PLANNER


def test_route_intent_with_scientific_general_goes_to_general(mock_llm):
    """GENERAL intent → GENERAL."""
    node = GeneralNodes(mock_llm, scientific_source=MagicMock())
    assert node.route_intent_with_scientific({'intent': Intent.GENERAL}) == NodeName.GENERAL


# --- 5c. router_agent data_source_scope ---

def test_router_sets_scope_both_for_multi_source(mock_llm):
    """LLM returns MULTI_SOURCE → data_source_scope == 'both'."""
    mock_runnable = MagicMock()
    mock_runnable.invoke.return_value = GeneralOrRagDecision(intent=Intent.MULTI_SOURCE)
    mock_llm.with_structured_output.return_value = mock_runnable

    node = GeneralNodes(mock_llm, scientific_source=MagicMock())
    state = {'messages': [HumanMessage(content="compare dataset with documents")]}
    result = node.router_agent(state)

    assert result['data_source_scope'] == 'both'
    assert result['intent'] == Intent.MULTI_SOURCE


def test_router_sets_scope_scientific_for_scientific(mock_llm):
    """LLM returns SCIENTIFIC → data_source_scope == 'scientific'."""
    mock_runnable = MagicMock()
    mock_runnable.invoke.return_value = GeneralOrRagDecision(intent=Intent.SCIENTIFIC)
    mock_llm.with_structured_output.return_value = mock_runnable

    node = GeneralNodes(mock_llm, scientific_source=MagicMock())
    state = {'messages': [HumanMessage(content="show NetCDF temperature variable")]}
    result = node.router_agent(state)

    assert result['data_source_scope'] == 'scientific'
    assert result['intent'] == Intent.SCIENTIFIC


def test_router_sets_scope_docs_for_rag(mock_llm):
    """LLM returns RAG → data_source_scope == 'docs'."""
    mock_runnable = MagicMock()
    mock_runnable.invoke.return_value = GeneralOrRagDecision(intent=Intent.RAG)
    mock_llm.with_structured_output.return_value = mock_runnable

    node = GeneralNodes(mock_llm)
    state = {'messages': [HumanMessage(content="What year did the drought start?")]}
    result = node.router_agent(state)

    assert result['data_source_scope'] == 'docs'
