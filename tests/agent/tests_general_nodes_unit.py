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
    """Positive Test: RAG intent goes to Query Rewriter"""
    state = {'intent': Intent.RAG}
    next_node = GeneralNodes.route_intent(state)
    assert next_node == NodeName.QUERY_REWRITER

def test_route_intent_logic_general():
    """Negative Test: General intent goes to General Node"""
    state = {'intent': Intent.GENERAL}
    next_node = GeneralNodes.route_intent(state)
    assert next_node == NodeName.GENERAL