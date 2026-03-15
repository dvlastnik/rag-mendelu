import pytest
import os
from pathlib import Path

from utils.logging_config import setup_logging
from generate_answers import generate_answers_file, get_results_filepath

def pytest_addoption(parser):
    parser.addoption(
        "--model", 
        action="store", 
        default="llama3.1:8b", 
        help="Name of the model to use for RAG"
    )

    parser.addoption(
        "--questions",
        action="store",
        default="tests/rag/questions/questions.json",
        help="Name of the file, that has questions on the dataset"
    )

    parser.addoption(
        "--collection-name",
        action="store",
        default="",
        help="Name of the Qdrant collection"
    )

    parser.addoption(
        "--regen", 
        action="store_true", 
        help="Force re-generation of RAG answers"
    )

@pytest.fixture(scope="session")
def model_name(request):
    """Returns the model name from CLI args."""
    return request.config.getoption("--model")

@pytest.fixture(scope="session")
def questions_file(request):
    """Returns the questions file path from CLI args."""
    return request.config.getoption("--questions")

def pytest_sessionstart(session):
    """
    Called after the Session object has been created and
    before performing collection and entering the run test loop.
    """
    setup_logging()
    model_name = session.config.getoption("--model")
    questions_file = session.config.getoption("--questions")
    collection_name = session.config.getoption("--collection-name")
    force_regen = session.config.getoption("--regen")
    
    base_dir = get_results_filepath(model_name, questions_file)
    results_file = Path(base_dir) / "answers.json"
    
    if force_regen or not results_file.exists():
        print(f"\n🔄 [SessionStart] Generating answers for {model_name}...")
        os.makedirs(base_dir, exist_ok=True)
        
        generate_answers_file(model_name, questions_file, collection_name)
        print("✅ [SessionStart] Generation Complete.\n")
    else:
        print(f"\n⚡ [SessionStart] Found existing answers at {results_file}. Skipping generation.\n")
