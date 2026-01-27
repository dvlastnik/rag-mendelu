import pytest
import json
import os
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.models import OllamaModel
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric

local_llm = OllamaModel(
    model="llama3.1:8b",
    base_url="http://localhost:11434",
    temperature=0
)

def load_evaluation_data():
    file_path = 'tests/rag/results/answers_260127_2027/answers.json'
    
    if not os.path.exists(file_path):
        pytest.fail(f"Test data file not found at: {file_path}")
        
    with open(file_path, "r") as f:
        # For now only 5 to test this out
        return json.load(f)[:5]

test_dataset = load_evaluation_data()

@pytest.mark.parametrize("data_item", test_dataset, ids=[f"ID_{i['id']}" for i in test_dataset])
def test_rag_evaluation(data_item):
    """
    Evaluates RAG performance using pre-generated answers from a JSON file.
    """
    
    question = data_item["question"]
    ground_truth = data_item["ground_truth"]
    actual_output = data_item["generated_answer"]
    
    raw_sources = data_item.get("retrieved_sources", "")

    # NOTE: If this is passed to LLMTestCase, then it takes a lot of time to finish, often times it fails, because local llm does not respond fast enough
    retrieval_context = [s.strip() for s in raw_sources.split("\n\n") if s.strip()]

    test_case = LLMTestCase(
        input=question,
        actual_output=actual_output,
        expected_output=ground_truth
    )

    relevancy = AnswerRelevancyMetric(
        threshold=0.5,
        model=local_llm,
        include_reason=True
    )
    
    faithfulness = FaithfulnessMetric(
        threshold=0.5,
        model=local_llm,
        include_reason=True
    )

    assert_test(test_case, [relevancy])