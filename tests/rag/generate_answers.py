import json
import time
import os
import sys
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List
from datetime import datetime

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root)

from database.base.MyDocument import MyDocument
from text_embedding import TextEmbeddingService
from database.QdrantDbRepository import QdrantDbRepository
from rag.AgenticRAG import AgenticRAG
import constants

def get_results_filepath(model_name: str, questions_file: str = '') -> str:
    current_timestamp = datetime.now().strftime("%y%m%d")
    if '/' in model_name:
        model_name = model_name.replace('/', '_')
    if ':' in model_name:
        model_name = model_name.replace(':', '_')

    if questions_file == '':
        return f'tests/rag/results/answers_{current_timestamp}_{model_name}'
    
    questions_path = Path(questions_file)
    return f'tests/rag/results/{questions_path.stem}_answers_{current_timestamp}_{model_name}'

def load_questions(filepath: str) -> List[Dict[str, any]]:
    with open(filepath, 'r') as f:
        return json.load(f)

def save_answers(results: List[Dict[str, any]], model_name: str, duration: float, questions_file: str):
    folder_path = get_results_filepath(model_name, questions_file=questions_file)
    output_path = f'{folder_path}/answers.json'
    os.makedirs(folder_path, exist_ok=True)

    output_data = {
        "metadata": {
            "model": model_name,
            "duration_seconds": round(duration, 2),
            "duration_minutes": round(duration / 60, 2),
            "total_questions": len(results)
        },
        "answers": results
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
def generate_anwers(questions: List[Dict[str, any]], model_name: str = 'llama3.1:8b', collection_name: str = '') -> List[Dict[str, any]]:
    embedding_service = TextEmbeddingService()

    valid_collection_name = constants.COLLECTION_NAME_DROUGH
    if collection_name != '':
        valid_collection_name = collection_name

    db_repository = QdrantDbRepository(
        ip='localhost', 
        port=6333,
        collection_name=valid_collection_name,
        metadata={
            'vector_size': int(os.environ.get("VECTOR_DB_VECTOR_SIZE", 384)),
            'distance': str(os.environ.get("VECTOR_DB_DISTANCE", "DOT"))
        }
    )
    connect_result = db_repository.connect()    
    if not connect_result.success:
        print(f"Failed to connect to ChromaDB for RAG: {connect_result.message}")
        return

    rag = AgenticRAG(database_service=db_repository, embedding_service=embedding_service, model_name=model_name)

    results = []
    for q in tqdm(questions, desc='Generating RAG Answers'):
        question = q['question']

        try:
            response = rag.chat(question)
            generated_text = response['response']
            original_query = response['original_query']
            rewritten_queries = response['rewritten_queries']

            distilled_facts = response.get('distilled_facts', [])

            agent_state = response.get('agent_state', {})
            completeness_follow_up_query = agent_state.get('completeness_follow_up_query', '')
            retrieval_iterations = agent_state.get('retrieval_iterations', 0)

            extracted_data = []
            for e in response['extracted_data']:
                extracted_data.append({
                    'year': e.year,
                    'location': e.location,
                    'entity': e.entities
                })

            sources = response['sources']
            retrieved_docs = []
            for source in sources:
                retrieved_docs.append({
                    'source': source.metadata['source'],
                    'text': source.text
                })
        except Exception as e:
            print(f"ERROR: {str(e)}")
            generated_text = f"ERROR: {str(e)}"
            retrieved_docs = []
            completeness_follow_up_query = ''
            retrieval_iterations = 0

        results.append({
            'id': q['id'],
            'question': question,
            'ground_truth': q['response'],
            'generated_answer': generated_text,
            'retrieved_sources': retrieved_docs,
            'distilled_facts': distilled_facts,
            'extracted_data': extracted_data,
            'original_query': original_query,
            'rewritten_queries': rewritten_queries,
            'completeness_follow_up_query': completeness_follow_up_query,
            'retrieval_iterations': retrieval_iterations
        })

    return results

def generate_answers_file(model_name: str, questions_filepath: str, collection_name: str):
    questions = load_questions(questions_filepath)

    start_time = time.time()
    results = generate_anwers(questions, model_name, collection_name)
    end_time = time.time()
    duration = end_time - start_time

    save_answers(results, model_name, duration, questions_filepath)