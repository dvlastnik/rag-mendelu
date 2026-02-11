import argparse
import os
import time
from dotenv import load_dotenv
import datetime
from pathlib import Path

from database.QdrantDbRepository import QdrantDbRepository
from database.base.BaseDbRepository import BaseDbRepository
from etl.DroughEtl import DroughEtl
from llm_handler.LLMHandler import LLMHandler
from metadata_extractor.graph import build_extractor_graph
from text_embedding_api.TextEmbeddingService import TextEmbeddingService
from rag.AgenticRAG import AgenticRAG
from utils.logging_config import get_logger, setup_logging, highlight_log
from utils.Utils import Utils
import constants

load_dotenv()

setup_logging()
logger = get_logger(__name__)

def run_etl_drough(path: str, delete_collection: bool, embedding_service: TextEmbeddingService, db_repository: BaseDbRepository, llm_handler: LLMHandler):
    """Runs the Drough (PDF) ETL pipeline on all files."""
    highlight_log(logger, "Starting Drough ETL pipeline...")
    if path != '':
        path_obj = Path(path)
        if path_obj.exists():
            if path_obj.is_dir():
                print('is a dir')
                pdf_files = Utils.find_files(folder_path=path, file_type='pdf')
            else:
                print('is a file')
                pdf_files = [path]
        else:
            raise FileNotFoundError(f'Folders or file at location {path_obj} not found')
    else:
        # TODO: Remove, this is only mocked this else branch should raise exception
        pdf_files = Utils.find_files(folder_path='/Users/david/Mendelu/Diplomka/drough_data/files', file_type='pdf')
        # raise FileNotFoundError(f'Folder path argument is not set: {folder_path}.')
        # pdf_files = ['/Users/david/Mendelu/Diplomka/drough_data/files/Provisional_State_of_the_Climate_2022_en.pdf']

    db_repository.collection_name = constants.COLLECTION_NAME_DROUGH
    db_repository.connect_and_create_collection(delete_collection)

    metadata_extractor = build_extractor_graph()

    for file in pdf_files:
        start_time = time.time()

        obj = DroughEtl(
                filepath=file,
                db_repositories={'qdrant': db_repository},
                embedding_service=embedding_service,
                metadata_extractor=metadata_extractor
            )
        status = obj.run()

        end_time = time.time()
        elapsed = end_time - start_time
        delta = datetime.timedelta(seconds=elapsed)
        highlight_log(logger, str(delta), character='~')
        
        if not status:
            logger.warning('ETL failed, stopping entire pipeline...')
            break
    highlight_log(logger, 'Drough ETL pipeline finished.')

def check_databases(db_repository: BaseDbRepository):
    """Checks the record counts for configured databases."""
    highlight_log(logger, "Checking database counts...")
    db_repository.collection_name = 'drough'

    db_repository.connect()
    db_repository.logger.info(f'Rows in drough: {db_repository.get_count()}')
    filenames = db_repository.get_all_filenames()
    db_repository.logger.info(f'Ingested files in db: {filenames} (size: {len(filenames)})')
    db_repository.logger.info(f'Metadata from database: {db_repository.valid_metadata}')
    db_repository.close()

def run_rag_chat(embedding_service: TextEmbeddingService, db_repository: BaseDbRepository):
    """
    Starts the RAG chat mode.
    """
    highlight_log(logger, "Starting RAG chat mode...")

    connect_result = db_repository.connect()    
    if not connect_result.success:
        logger.error(f"Failed to connect to ChromaDB for RAG: {connect_result.message}")
        return

    rag = AgenticRAG(database_service=db_repository, embedding_service=embedding_service)

    # question = "The report explicitly states that the Greenland Ice Sheet lost approximately 85 Gt of ice in 2022. What was the specific total mass balance loss (in Gigatonnes) for the Antarctic Ice Sheet reported for the same period?"
    print("Assitant ready. Type 'exit' to end program.")
    while True:
        question = input('Enter question: ')
        if 'exit' in question.lower():
            break

        result = rag.chat(question)
        print(f'Assistant: {result['response']}')
        print('//////////////////////////////////')
        for index, source in enumerate(result['sources']):
            print(f'--- Source {index}: ---')
            print(f'Source from file: {source.metadata['source']}')
            print(f' Source Text: {source.text}')
            print(f'-----------------------')
        print('//////////////////////////////////')

def parse_args():
    parser = argparse.ArgumentParser(description='RAG app')

    parser.add_argument(
        '--vector-db',
        type=str,
        default='chroma',
        choices=['chroma', 'qdrant'],
        help='Type of vector database to use'
    )

    parser.add_argument(
        '--run-etl',
        action='store_true',
        help='Run a ETL'
    )

    parser.add_argument(
        '--path',
        type=str,
        default='',
        help='Path to folder, where pdf files are stored'
    )

    parser.add_argument(
        '--check-dbs',
        action='store_true',
        help='Check the status and count of all databases.'
    )

    parser.add_argument(
        '--erase-db',
        action='store_true',
        help='If to erase database before starting ETL'
    )

    parser.add_argument(
        '--chat',
        action='store_true',
        help='Rag CHAT'
    )

    return parser.parse_args()

def main():
    start_time = time.time()
    args = parse_args()

    # Objects
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
    llm_handler = LLMHandler(ip='localhost', port=11434)

    if args.run_etl:
        run_etl_drough(args.path, args.erase_db, embedding_service, db_repository, llm_handler)
        
    elif args.check_dbs:
        check_databases(db_repository)
    elif args.chat:
        run_rag_chat(embedding_service, db_repository)
    else:
        logger.info("No ETL or DB check specified. Running RAG chat mode by default.")
        run_rag_chat(embedding_service, db_repository)

    # Close dbs
    db_repository.close()
    
    end_time = time.time()
    elapsed = end_time - start_time
    delta = datetime.timedelta(seconds=elapsed)
    highlight_log(logger, str(delta), character='~')
    
if __name__ == '__main__':
    main()
    