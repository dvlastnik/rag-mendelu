import pytest
import os
from pathlib import Path
from generate_answers import generate_answers, get_results_filepath

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
        default="tests/rag/questions.json",
        help="Name of the file, that has questions on the dataset"
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
    # Access CLI args via session.config
    model_name = session.config.getoption("--model")
    questions_file = session.config.getoption("--questions")
    force_regen = session.config.getoption("--regen")
    
    # Check if we need to generate
    # Note: We use the helper to get the folder, then append the filename
    base_dir = get_results_filepath(model_name)
    results_file = Path(base_dir) / "answers.json"
    
    if force_regen or not results_file.exists():
        print(f"\n🔄 [SessionStart] Generating answers for {model_name}...")
        # Ensure directory exists just in case
        os.makedirs(base_dir, exist_ok=True)
        
        # Run your generation script
        generate_answers(model_name, questions_file)
        print("✅ [SessionStart] Generation Complete.\n")
    else:
        print(f"\n⚡ [SessionStart] Found existing answers at {results_file}. Skipping generation.\n")

# @pytest.fixture(scope="session", autouse=True)
# def ensure_answers_generated(request, model_name, questions_file):
#     """
#     Runs BEFORE any test starts.
#     Checks if answers exist. If not (or if --regen flag is used), generates them.
#     """
#     force_regen = request.config.getoption("--regen")
#     results_path = Path(get_results_filepath)
#     should_generate = force_regen or not results_path.exists()
#     if should_generate:
#         print("\n🔄 [Setup] Generating new RAG answers (this may take a while)...")
#         print(f"   [Setup] Using model: {model_name}, questions file: {questions_file}")
#         generate_answers(model_name, questions_file)
#         print("✅ [Setup] DONE")
#     else:
#         print(f"\n⚡ [Setup] SKIPPING generation. Found existing: {results_path}")
#         print("   (Use --regen to force new answers)")