import pytest
import json
import os

from generate_answers import get_results_filepath
from tests.rag.judge import Judge

session_results = []

def pytest_generate_tests(metafunc):
    """
    This special Pytest hook runs at 'Collection Time'.
    It allows us to read CLI args (--model) and generate tests dynamically.
    """
    if "data" in metafunc.fixturenames:
        model_name = metafunc.config.getoption("--model")
        base_path = get_results_filepath(model_name) 
        file_path = os.path.join(base_path, "answers.json")
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                dataset = json.load(f)['answers']
        else:
            print(f"⚠️ Warning: {file_path} not found during collection.")
            dataset = []

        metafunc.parametrize(
            'data', 
            dataset, 
            ids=[f"ID_{item['id']}" for item in dataset]
        )

@pytest.fixture(scope="session")
def judge():
    """
    Initializes the Judge using the model name passed via CLI (--model).
    """
    return Judge()

@pytest.fixture(scope="session", autouse=True)
def evaluation_logger(model_name, questions_file):
    """
    This fixture runs once per session.
    It waits for all tests to finish, then saves 'session_results' to a file.
    """
    yield
    
    if session_results:
        output_dir = get_results_filepath(model_name, questions_file)
        os.makedirs(output_dir, exist_ok=True)
        
        if '/' in model_name:
            model_name = model_name.replace('/', '_')
        output_path = os.path.join(output_dir, f'judgement_report_{model_name}.json')
        answers_path = os.path.join(output_dir, 'answers.json')

        duration = 0
        if os.path.exists(answers_path):
            with open(answers_path, 'r') as f:
                data = json.load(f)
                duration = data.get('metadata', {}).get('duration_seconds', 0)
        
        total = len(session_results)
        passed = sum(1 for r in session_results if r['status'] == 'PASSED')
        accuracy = (passed / total) * 100 if total > 0 else 0
        
        final_report = {
            'metadata': {
                'duration_seconds': round(duration, 2),
                'duration_minutes': round(duration / 60, 2),
                'total_tests': total,
                'passed': passed,
                'failed': total - passed,
                'accuracy_percent': round(accuracy, 2)
            },
            'details': session_results
        }
        
        with open(output_path, "w") as f:
            json.dump(final_report, f, indent=2)

def test_rag_quality(data, judge):
    raw_sources = data.get("retrieved_sources", "")
    retrieval_context = [s.strip() for s in raw_sources.split("\n\n") if s.strip()]
    
    eval_result = judge.evaluate(
        question=data['question'],
        answer=data['generated_answer'],
        context=retrieval_context,
        ground_truth=data['ground_truth']
    )
    
    pass_str = 'PASSED'
    fail_str = 'FAILED'
    status_str = fail_str
    if eval_result.relevancy_score >= 4 and eval_result.faithfulness_score == 5:
        status_str = 'PASSED'

    log_entry = {
        'id': data['id'],
        'question': data['question'],
        'answer': data['generated_answer'],
        'extracted_data': data['extracted_data'],
        'retrieved_sources': data['retrieved_sources'],
        'rewritten_query': data['rewritten_query'],
        'true_answer': data['ground_truth'],
        'scores': {
            'relevancy': eval_result.relevancy_score,
            'faithfulness': eval_result.faithfulness_score
        },
        'reasoning': eval_result.reasoning,
        'status': status_str
    }
    session_results.append(log_entry)

    print(f"\n[ID {data['id']}] Status: {status_str} | Rel: {eval_result.relevancy_score} | Faith: {eval_result.faithfulness_score}")
    print(f"Reason: {eval_result.reasoning}")

    error_msg = f"Judge Failed this answer. Reason: {eval_result.reasoning}"
    assert status_str is pass_str, error_msg