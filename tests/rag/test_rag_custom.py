import pytest
import json
import os
from datetime import datetime
from tests.rag.judge import Judge

judge = Judge()
session_results = []

def load_test_data():
    path = 'tests/rag/results/answers_260127_2027/answers.json'
    if not os.path.exists(path):
        pytest.fail(f"Data file not found: {path}")
    with open(path, "r") as f:
        # For now only 5 to test this out
        return json.load(f)

test_dataset = load_test_data()

# --- FIXTURE: LOGGING ---
@pytest.fixture(scope="session", autouse=True)
def evaluation_logger():
    """
    This fixture runs once per session.
    It waits for all tests to finish, then saves 'session_results' to a file.
    """
    yield
    
    if session_results:
        output_dir = "tests/rag/results/answers_260127_2027"
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f"judgement_report.json"
        output_path = os.path.join(output_dir, filename)
        
        total = len(session_results)
        passed = sum(1 for r in session_results if r['status'] == 'PASSED')
        accuracy = (passed / total) * 100 if total > 0 else 0
        
        final_report = {
            "meta": {
                "total_tests": total,
                "passed": passed,
                "failed": total - passed,
                "accuracy_percent": round(accuracy, 2)
            },
            "details": session_results
        }
        
        with open(output_path, "w") as f:
            json.dump(final_report, f, indent=2)

@pytest.mark.parametrize("data", test_dataset, ids=[f"ID_{i['id']}" for i in test_dataset])
def test_rag_quality(data):
    raw_sources = data.get("retrieved_sources", "")
    retrieval_context = [s.strip() for s in raw_sources.split("\n\n") if s.strip()]
    
    eval_result = judge.evaluate(
        question=data['question'],
        answer=data['generated_answer'],
        context=retrieval_context,
        ground_truth=data['ground_truth']
    )
    
    log_entry = {
        "id": data['id'],
        "question": data['question'],
        "answer": data['generated_answer'],
        "true_answer": data['ground_truth'],
        "scores": {
            "relevancy": eval_result.relevancy_score,
            "faithfulness": eval_result.faithfulness_score
        },
        "pass_fail": eval_result.pass_fail,
        "reasoning": eval_result.reasoning,
        "status": "PASSED" if eval_result.pass_fail else "FAILED"
    }
    session_results.append(log_entry)

    print(f"\n[ID {data['id']}] Pass: {eval_result.pass_fail} | Rel: {eval_result.relevancy_score} | Faith: {eval_result.faithfulness_score}")
    print(f"Reason: {eval_result.reasoning}")

    error_msg = f"Judge Failed this answer. Reason: {eval_result.reasoning}"
    assert eval_result.pass_fail is True, error_msg