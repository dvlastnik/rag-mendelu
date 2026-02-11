import json
import time
import os
import sys
from tqdm import tqdm
from typing import Dict, List
from datetime import datetime

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root)

from text_embedding_api.TextEmbeddingService import TextEmbeddingService
from database.QdrantDbRepository import QdrantDbRepository
from database.base.BaseDbRepository import BaseDbRepository
from rag.AgenticRAG import AgenticRAG
import constants

def get_results_filepath(model_name: str) -> str:
    current_timestamp = datetime.now().strftime("%y%m%d")
    if '/' in model_name:
        model_name = model_name.replace('/', '_')
    return f'tests/rag/results/answers_{current_timestamp}_{model_name}'

def load_questions(filepath: str) -> List[Dict[str, any]]:
    with open(filepath, 'r') as f:
        return json.load(f)

def save_answers(results: List[Dict[str, any]], model_name: str, duration: float):
    folder_path = get_results_filepath(model_name)
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
    
def generate_anwers(questions: List[Dict[str, any]], model_name: str = 'llama3.1:8b') -> List[Dict[str, any]]:
    embedding_service = TextEmbeddingService(ip="localhost", port=8000)
    db_repository = QdrantDbRepository(
        ip='localhost', 
        port=6333,
        collection_name=constants.COLLECTION_NAME_DROUGH,
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
            state = response['agent_state']
            original_query = state['original_query']
            rewritten_query = state['final_query']

            expanded_queries = []
            extracted_data = []
            for e in state['extracted_data']:
                extracted_data.append({
                    'year': e.year,
                    'location': e.location,
                    'topics': e.topics
                })

                topics_str = ', '.join(e.topics)
                expanded_queries.append(f'{rewritten_query}\n{topics_str}')

            sources = response['sources']
            retrieved_docs = ''
            for source in sources:
                retrieved_docs += f'Source from file: {source.metadata['source']}\nSource Text: {source.text}\n\n'
        except Exception as e:
            generated_text = f"ERROR: {str(e)}"
            retrieved_docs = []

        results.append({
            'id': q['id'],
            'question': question,
            'ground_truth': q['response'],
            'generated_answer': generated_text,
            'retrieved_sources': retrieved_docs,
            'extracted_data': extracted_data,
            'original_query': original_query,
            'final_query': rewritten_query,
            'expanded_queries': expanded_queries
        })

    return results

def generate_answers_file(model_name: str, questions_filepath: str):
    questions = load_questions(questions_filepath)

    start_time = time.time()
    results = generate_anwers(questions, model_name)
    end_time = time.time()
    duration = end_time - start_time

    save_answers(results, model_name, duration)