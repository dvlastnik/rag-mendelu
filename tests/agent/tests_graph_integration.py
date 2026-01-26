import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage

from rag.agents.enums import Intent
from rag.agents.models import (
    GeneralOrRagDecision, 
    MultiExtraction, 
    GradeDocuments, 
    GradeHallucinations, 
    ExtractionScheme 
)
from rag.agents.graph import build_graph 

@pytest.fixture
def mock_deps():
    """Mocks the external DB and Embedding Service."""
    db = MagicMock()
    search_res = MagicMock()
    search_res.success = True
    search_res.data = [MagicMock(text="Brno Floods Data")]
    db.search.return_value = search_res

    emb = MagicMock()
    emb_data = MagicMock()
    emb_data.embedding = [0.1] * 10
    emb_data.sparse = {}
    emb.get_embedding_with_uuid.return_value = [emb_data]
    
    return db, emb

def setup_mock_llm():
    mock_llm = MagicMock()
    
    mock_llm.invoke.side_effect = [
        AIMessage(content="Rewritten Query: Floods in Brno"), 
        AIMessage(content="Final Answer: There were floods in Brno."), 
    ]

    def structured_side_effect(schema, **kwargs):
        mock_runnable = MagicMock()
        
        schema_name = getattr(schema, "__name__", str(schema))

        if "GeneralOrRagDecision" in schema_name:
            mock_runnable.invoke.return_value = GeneralOrRagDecision(intent=Intent.RAG)
        
        elif "MultiExtraction" in schema_name:
            t = ExtractionScheme(
                city="Brno", 
                year="2024", 
                topics=["floods"], 
                country="Czechia"
            )
            mock_runnable.invoke.return_value = MultiExtraction(targets=[t])
            
        elif "GradeDocuments" in schema_name:
            mock_runnable.invoke.return_value = GradeDocuments(is_relevant="yes")
            
        elif "GradeHallucinations" in schema_name:
            mock_runnable.invoke.return_value = GradeHallucinations(is_relevant="yes")
            
        return mock_runnable

    mock_llm.with_structured_output.side_effect = structured_side_effect
    return mock_llm

@patch("rag.agents.graph.init_chat_model")
def test_integration_rag_happy_path(mock_init, mock_deps):
    mock_db, mock_emb = mock_deps
    mock_llm = setup_mock_llm()
    mock_init.return_value = mock_llm

    app = build_graph(mock_db, mock_emb)

    inputs = {"messages": [HumanMessage(content="Tell me about Brno floods")]}
    final_state = app.invoke(inputs)

    assert "Final Answer: There were floods in Brno." in final_state['messages'][-1].content
    assert len(final_state['extracted_data']) == 1
    assert final_state['extracted_data'][0].city == "Brno"
    assert final_state['hallucination_status'] == 'clean'

@patch("rag.agents.graph.init_chat_model")
def test_integration_rag_extraction_failure(mock_init, mock_deps):
    mock_db, mock_emb = mock_deps
    mock_llm = MagicMock()
    
    mock_llm.invoke.return_value = AIMessage(content="Vague Query")

    def structured_side_effect(schema, **kwargs):
        mock_runnable = MagicMock()
        if schema == GeneralOrRagDecision:
            mock_runnable.invoke.return_value = GeneralOrRagDecision(intent=Intent.RAG)
        elif schema == MultiExtraction:
            # RETURN EMPTY TARGETS
            mock_runnable.invoke.return_value = MultiExtraction(targets=[])
        return mock_runnable

    mock_llm.with_structured_output.side_effect = structured_side_effect
    mock_init.return_value = mock_llm

    app = build_graph(mock_db, mock_emb)
    
    inputs = {"messages": [HumanMessage(content="Blah blah random text")]}
    final_state = app.invoke(inputs)

    last_msg = final_state['messages'][-1].content
    assert "I could not process your request" in last_msg