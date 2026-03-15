import pytest
from unittest.mock import MagicMock

from rag.agents.nodes.rag_nodes import RagNodes

@pytest.fixture
def mock_llm():
    return MagicMock()

@pytest.fixture
def mock_db():
    repo = MagicMock()
    mock_result = MagicMock()
    mock_result.success = True
    mock_result.data = [MagicMock(text="Doc Content")]
    repo.search.return_value = mock_result
    return repo

@pytest.fixture
def mock_embedding():
    service = MagicMock()
    mock_data = MagicMock()
    mock_data.embedding = [0.1, 0.2]
    mock_data.sparse = {"token": 1}
    service.get_embedding_with_uuid.return_value = [mock_data]
    return service

@pytest.fixture
def rag_nodes(mock_llm, mock_db, mock_embedding):
    return RagNodes(mock_llm, mock_db, mock_embedding)