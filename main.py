import argparse
import os
import time
from dotenv import load_dotenv
import datetime
from pathlib import Path

from database.qdrant_db_repository import QdrantDbRepository
from database.base.base_db_repository import BaseDbRepository
from database.duck_db_repository import DuckDbRepository
from etl.general_etl import GeneralEtl
from text_embedding import TextEmbeddingService
from rag.agentic_rag import AgenticRAG
from utils.logging_config import get_logger, setup_logging, highlight_log

load_dotenv()

setup_logging()
logger = get_logger(__name__)

def run_etl_general(
        path: str,
        delete_collection: bool,
        embedding_service: TextEmbeddingService,
        db_repository: BaseDbRepository,
        collection_name: str,
        use_recursive_chunking: bool,
        duck_db=None,
    ):
    """Runs the general ETL pipeline on any supported file type."""
    highlight_log(logger, "Starting General ETL pipeline...")

    if not path:
        raise ValueError("--path is required for ETL.")

    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Path not found: {path_obj}")

    if path_obj.is_dir():
        files = [
            str(f) for f in path_obj.rglob("*")
            if f.is_file() and f.suffix.lower() in GeneralEtl.SUPPORTED_EXTENSIONS
        ]
        logger.info(f"Found {len(files)} supported files in '{path_obj}'")
    else:
        files = [str(path_obj)]

    if delete_collection and duck_db is not None:
        for file in files:
            duck_db.drop_table(Path(file).stem)

    if collection_name:
        db_repository.collection_name = collection_name
    db_repository.connect_and_create_collection(delete_collection)

    use_semantic = not use_recursive_chunking

    for file in files:
        start_time = time.time()

        obj = GeneralEtl(
            filepath=file,
            db_repositories={'qdrant': db_repository},
            embedding_service=embedding_service,
            use_semantic=use_semantic,
            duck_db_repo=duck_db,
        )
        status = obj.run()

        elapsed = time.time() - start_time
        highlight_log(logger, str(datetime.timedelta(seconds=elapsed)), character='~')

        if not status:
            logger.warning('ETL failed, stopping entire pipeline...')
            break

    highlight_log(logger, 'General ETL pipeline finished.')

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

def run_rag_chat(embedding_service: TextEmbeddingService, db_repository: BaseDbRepository, model_name: str, duck_db: DuckDbRepository | None = None):
    """
    Starts the RAG chat mode.
    """
    highlight_log(logger, "Starting RAG chat mode...")

    connect_result = db_repository.connect()    
    if not connect_result.success:
        logger.error(f"Failed to connect to ChromaDB for RAG: {connect_result.message}")
        return

    rag = AgenticRAG(database_service=db_repository, embedding_service=embedding_service, model_name=model_name, duck_db_repo=duck_db)

    print("Assitant ready. Type 'exit' or press Ctrl+C to end program.")
    while True:
        try:
            question = input('Enter question: ')
            if 'exit' in question.lower():
                break

            result = rag.chat(question)
            print('//////////////////////////////////')
            for index, source in enumerate(result['sources']):
                print(f'--- Source {index}: ---')
                print(f'Source from file: {source.metadata['source']}')
                print(f' Source Text: {source.text}')
                print(f'-----------------------')
            print('//////////////////////////////////')
            for index, facts_block in enumerate(result['distilled_facts']):
                print(f'Distilled facts [{index}]: {facts_block}')
            print('----------------------------------')
            print(f'Assistant: {result['response']}')
            print('----------------------------------')
        except KeyboardInterrupt:
            break

def parse_args():
    parser = argparse.ArgumentParser(description='RAG app')

    parser.add_argument(
        '--model',
        type=str,
        default='llama3.1:8b',
        help='LLM name'
    )

    parser.add_argument(
        '--run-etl',
        action='store_true',
        help='Run a ETL'
    )

    parser.add_argument(
        '--collection-name',
        type=str,
        default='',
        help='Name of the Qdrant collection'
    )

    parser.add_argument(
        '--recursive-chunking',
        action='store_true',
        help='Whether to use recursive chunking'
    )

    parser.add_argument(
        '--embed-model',
        type=str,
        default=None,
        help=(
            'Dense embedding model name. '
            'If the model is in fastembed\'s supported list it is used via fastembed; '
            'otherwise sentence_transformers is tried (downloads from HuggingFace). '
            'Defaults to BAAI/bge-small-en-v1.5 (fastembed).'
        )
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
        '--erase',
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
    embedding_service = TextEmbeddingService(dense_model=args.embed_model)
    vector_size = embedding_service.get_embedding_dim()
    logger.info(
        f"Embedding model: {embedding_service.get_current_model()} "
        f"(library: {embedding_service.get_library()}, dim: {vector_size})"
    )

    collection_name = os.environ.get("COLLECTION_NAME", "default_name")
    if args.collection_name:
        collection_name = args.collection_name

    db_repository = QdrantDbRepository(
        ip='localhost',
        port=6333,
        collection_name=collection_name,
        metadata={
            'vector_size': vector_size,
            'distance': str(os.environ.get("VECTOR_DB_DISTANCE", "DOT"))
        }
    )

    duck_db = DuckDbRepository()

    if args.run_etl:
        run_etl_general(
            path=args.path,
            delete_collection=args.erase,
            embedding_service=embedding_service,
            db_repository=db_repository,
            collection_name=collection_name,
            use_recursive_chunking=args.recursive_chunking,
            duck_db=duck_db,
        )
        
    elif args.check_dbs:
        check_databases(db_repository)
    elif args.chat:
        run_rag_chat(embedding_service, db_repository, args.model, duck_db=duck_db)
    else:
        logger.info("No ETL or DB check specified. Running RAG chat mode by default.")
        run_rag_chat(embedding_service, db_repository, args.model, duck_db=duck_db)

    # Close dbs
    duck_db.close()
    db_repository.close()
    
    end_time = time.time()
    elapsed = end_time - start_time
    delta = datetime.timedelta(seconds=elapsed)
    highlight_log(logger, str(delta), character='~')
    
if __name__ == '__main__':
    main()
    