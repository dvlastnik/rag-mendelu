import argparse
import json
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
from tui.tui import TuiWizard
from tui.chat import TuiChat

load_dotenv()

setup_logging()
logger = get_logger(__name__)

def run_etl_general(
        path: str,
        delete_collection: bool,
        embedding_service: TextEmbeddingService,
        db_repository: BaseDbRepository,
        collection_name: str,
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

    succeeded = []
    failed = []

    print(f"\nIngesting {len(files)} file(s) into collection '{collection_name}'...\n")

    for file in files:
        print(f"  -> {Path(file).name}")
        start_time = time.time()

        obj = GeneralEtl(
            filepath=file,
            db_repositories={'qdrant': db_repository},
            embedding_service=embedding_service,
            duck_db_repo=duck_db,
        )
        status = obj.run()

        elapsed = time.time() - start_time
        highlight_log(logger, str(datetime.timedelta(seconds=elapsed)), character='~')

        if not status:
            print(f"  x Failed:  {Path(file).name}")
            logger.warning('ETL failed for %s, skipping...', Path(file).name)
            failed.append(file)
            continue

        print(f"  v Done:    {Path(file).name}  ({datetime.timedelta(seconds=int(elapsed))})\n")
        succeeded.append(file)

    print(f"ETL complete — {len(succeeded)} succeeded, {len(failed)} failed.")
    highlight_log(logger, 'General ETL pipeline finished.')
    logger.info('ETL Summary — succeeded (%d): %s', len(succeeded), [Path(f).name for f in succeeded])
    if failed:
        logger.warning('ETL Summary — failed (%d): %s', len(failed), [Path(f).name for f in failed])

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

def _format_result_json(result: dict) -> str:
    """Serializes a rag.chat() result dict to a JSON string."""
    sources = []
    for doc in result.get('sources') or []:
        sources.append({
            'text': doc.text,
            'score': doc.score,
            'metadata': {k: v for k, v in (doc.metadata or {}).items()
                         if k not in ('embedding', 'sparse_embedding')},
        })

    agent_state = result.get('agent_state') or {}
    sql_result = agent_state.get('sql_result')

    payload = {
        'response': result.get('response', ''),
        'sources': sources,
        'distilled_facts': result.get('distilled_facts') or [],
    }
    if sql_result:
        payload['sql_result'] = sql_result

    return json.dumps(payload, ensure_ascii=False, indent=2)


def _print_result(result: dict):
    """Prints a rag.chat() result in human-readable format."""
    print('//////////////////////////////////')
    for index, source in enumerate(result['sources']):
        print(f'--- Source {index}: ---')
        print(f'Source from file: {source.metadata["source"]}')
        print(f' Source Text: {source.text}')
        print(f'-----------------------')
    print('//////////////////////////////////')
    for index, facts_block in enumerate(result['distilled_facts']):
        print(f'Distilled facts [{index}]: {facts_block}')
    print('----------------------------------')
    print(f'Assistant: {result["response"]}')
    print('----------------------------------')


def run_rag_chat(rag: AgenticRAG, json_output: bool = False):
    """
    Starts the RAG chat mode.
    """
    highlight_log(logger, "Starting RAG chat mode...")
    print("Assitant ready. Type 'exit' or press Ctrl+C to end program.")
    while True:
        try:
            question = input('Enter question: ')
            if 'exit' in question.lower():
                break

            result = rag.chat(question)
            if json_output:
                print(_format_result_json(result))
            else:
                _print_result(result)
        except KeyboardInterrupt:
            break

def ask_rag(question: str, rag: AgenticRAG, json_output: bool = False):
    """
    Only sends one question to rag which will return response and then exit.
    """
    highlight_log(logger, f"Asking RAG question: '{question}'")
    try:
        result = rag.chat(question)
        if json_output:
            print(_format_result_json(result))
        else:
            _print_result(result)
    except KeyboardInterrupt:
        pass

def parse_args():
    parser = argparse.ArgumentParser(description='RAG app')

    parser.add_argument(
        '--model',
        type=str,
        default='ministral-3:8b',
        help='LLM name'
    )

    parser.add_argument(
        '--collection-name',
        type=str,
        default='',
        help='Name of the Qdrant collection'
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
        '--erase',
        action='store_true',
        help='If to erase database before starting ETL'
    )

    parser.add_argument(
        '--json-output',
        action='store_true',
        help='Output responses as JSON (only with --chat or --ask)'
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        '--run-etl',
        action='store_true',
        help='Run a ETL'
    )
    mode_group.add_argument(
        '--chat',
        action='store_true',
        help='Rag CHAT, interactive'
    )
    mode_group.add_argument(
        '--ask',
        type=str,
        default='',
        help='Only ask one question and get answer, NOT interactive'
    )
    mode_group.add_argument(
        '--check-dbs',
        action='store_true',
        help='Check the status and count of all databases.'
    )

    args = parser.parse_args()

    if args.erase and not args.run_etl:
        parser.error('--erase can only be used with --run-etl')
    if args.path and not args.run_etl:
        parser.error('--path can only be used with --run-etl')
    if args.json_output and not (args.chat or args.ask):
        parser.error('--json-output can only be used with --chat or --ask')

    return args

def main():
    start_time = time.time()
    args = parse_args()

    if not any([args.run_etl, args.check_dbs, args.ask, args.chat]):
        args = TuiWizard(
            default_model=os.environ.get("OLLAMA_MODEL", "ministral-3:8b"),
            qdrant_host=os.environ.get("QDRANT_HOST", "localhost"),
            qdrant_port=int(os.environ.get("QDRANT_PORT", "6333")),
        ).run()
        setup_logging(silent_console=True, log_file=args.log_file)

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
        ip=os.environ.get("QDRANT_HOST", "localhost"),
        port=int(os.environ.get("QDRANT_PORT", "6333")),
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
            duck_db=duck_db,
        )
    elif args.check_dbs:
        check_databases(db_repository)
    else:
        rag = AgenticRAG(
            database_service=db_repository,
            embedding_service=embedding_service,
            model_name=args.model,
            duck_db_repo=duck_db,
        )
        if getattr(args, 'tui_mode', False):
            if args.ask:
                TuiChat(rag, args.model).run_ask(args.ask)
            else:
                TuiChat(rag, args.model).run_chat()
        elif args.ask:
            ask_rag(args.ask, rag, json_output=args.json_output)
        else:
            run_rag_chat(rag, json_output=args.json_output)

    duck_db.close()
    db_repository.close()
    
    end_time = time.time()
    elapsed = end_time - start_time
    delta = datetime.timedelta(seconds=elapsed)
    highlight_log(logger, str(delta), character='~')
    
if __name__ == '__main__':
    main()
    