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
def mock_sql_db():
    repo = MagicMock()
    repo.get_compact_catalog.return_value = "games_2025(name:varchar, review:double) [20 rows]"
    repo.list_tables.return_value = ["games_2025"]
    repo.get_schema.return_value = "Table: games_2025\n\nColumns:\n name  varchar\nreview   double\n\nSample rows:\n name  review\n Game1     9.4"
    repo.run_select.return_value = MagicMock(empty=False, to_string=lambda index: "name  review\nGame1    9.4")
    return repo

@pytest.fixture
def rag_nodes(mock_llm, mock_db, mock_embedding, mock_sql_db):
    return RagNodes(mock_llm, mock_db, mock_embedding, sql_db_repo=mock_sql_db)