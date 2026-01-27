import json
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

def load_questions() -> List[Dict[str, any]]:
    with open('tests/rag/questions.json', 'r') as f:
        return json.load(f)
    
def save_answers(results: List[Dict[str, any]]):
    current_timestamp = datetime.now().strftime("%y%m%d_%H%M")

    folder_path = f'tests/rag/results/answers_{current_timestamp}'
    output_path = f'{folder_path}/answers.json'
    os.makedirs(folder_path, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
def generate_anwers(questions: List[Dict[str, any]]) -> List[Dict[str, any]]:
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

    rag = AgenticRAG(database_service=db_repository, embedding_service=embedding_service)

    results = []
    for q in tqdm(questions, desc='Generating RAG Answers'):
        question = q['question']

        try:
            response = rag.chat(question)
            generated_text = response['response']
            sources = response['sources']
            retrieved_docs = ''
            for source in sources:
                retrieved_docs += f'Source from file: {source.metadata['source']}\nSource Text: {source.text}\n\n'
        except Exception as e:
            generated_text = f"ERROR: {str(e)}"
            retrieved_docs = []

        results.append({
            "id": q['id'],
            "question": question,
            "ground_truth": q['response'],
            "generated_answer": generated_text,
            "retrieved_sources": retrieved_docs
        })

    return results

def main():
    questions = load_questions()
    results = generate_anwers(questions)
    save_answers(results)
    
if __name__ == '__main__':
    main()