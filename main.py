import argparse
from database.ChromaDbRepository import ChromaDbRepository
from database.base.DbRepositoryFactory import DbRepositoryFactory
from database.base.BaseDbRepository import BaseDbRepository
from etl.DroughEtl import DroughEtl
from text_embedding_api.TextEmbeddingService import TextEmbeddingService
from etl.Mzdr1DataEtl import Mzdr1DataEtl
from rag.RAG import RAG
from utils.logging_config import get_logger, setup_logging, highlight_log
from utils.Utils import Utils
import constants

setup_logging()
logger = get_logger(__name__)

def run_etl_csu(embedding_service: TextEmbeddingService, db_repository: BaseDbRepository):
    """Runs the CSU (Mzdr1DataEtl) ETL pipeline."""
    highlight_log(logger, "Starting ETL pipeline CSU")

    db_repository.collection_name = constants.COLLECTION_NAME_CSU
    db_repository.metadata=constants.CHROMA_DB_METADATA_CSU
    
    etl = Mzdr1DataEtl(
        filepath="data/csu/MZDR_1_data.csv", 
        db_repository=db_repository,
        embedding_service=embedding_service
    )
    etl.run()
    highlight_log(logger, "CSU ETL pipeline finished.")

def run_etl_drough(embedding_service: TextEmbeddingService, db_repository: BaseDbRepository):
    """Runs the Drough (PDF) ETL pipeline on all files."""
    highlight_log(logger, "Starting Drough ETL pipeline...")
    pdf_files = Utils.find_files(folder_path='/Users/david/Mendelu/Diplomka/drough_data/files', file_type='pdf')

    db_repository.collection_name = constants.COLLECTION_NAME_DROUGH
    db_repository.metadata = constants.CHROMA_DB_METADATA_DROUGH
    db_repository.connect_and_create_collection()

    for file in pdf_files:
        obj = DroughEtl(
                filepath=file,
                db_repository=db_repository,
                embedding_service=embedding_service
            )
        obj.run()
    highlight_log(logger, "Drough ETL pipeline finished.")

def check_databases():
    """Checks the record counts for configured databases."""
    highlight_log(logger, "Checking database counts...")
    # TODO: Create script that will accept some file, load it and will check all dbs that are in the config file
    # so it will not be hardcoded
    # BaseDbRepository.check_count_for(ChromaDbRepository, ip="localhost", port=8001, collection_name=constants.COLLECTION_NAME_CSU, metadata=constants.CHROMA_DB_METADATA_CSU)
    BaseDbRepository.check_count_for(ChromaDbRepository, ip="localhost", port=8001, collection_name=constants.COLLECTION_NAME_DROUGH, metadata=constants.CHROMA_DB_METADATA_DROUGH)


def run_rag_chat(embedding_service: TextEmbeddingService):
    """
    Starts the RAG chat mode.
    """
    highlight_log(logger, "Starting RAG chat mode...")
    chroma = ChromaDbRepository(ip="localhost", port=8001, collection_name=constants.COLLECTION_NAME_CSU, metadata=constants.CHROMA_DB_METADATA_CSU)
    connect_result = chroma.connect(create_collection=False)
    
    if not connect_result.success:
        logger.error(f"Failed to connect to ChromaDB for RAG: {connect_result.message}")
        return

    rag = RAG(database_service=chroma, embedding_service=embedding_service, model_path="/Users/david/Library/Application Support/nomic.ai/GPT4All/Meta-Llama-3.1-8B-Instruct-128k-Q4_0.gguf")

    question = "Jaký byl průměrný evidenční počet zaměstnanců přepočtený (tis. osob) v sektoru F - Stavebnictví v roce 2000?"
    logger.info(f"Chatting with question: {question}")
    rag.chat(question)

def parse_args():
    parser = argparse.ArgumentParser(description="Main app")

    parser.add_argument(
        '--vector-db',
        type=str,
        default='chroma',
        choices=['chroma', 'qdrant'], # TODO: Add other
        help="Type of vector database to use"
    )

    parser.add_argument(
        '--model-path',
        type=str,
        default='/Users/david/Library/Application Support/nomic.ai/GPT4All/Meta-Llama-3.1-8B-Instruct-128k-Q4_0.gguf',
        help="Path to llm model (.gguf)"
    )

    parser.add_argument(
        '--run-etl',
        type=str,
        choices=['csu', 'drough'], # Add another options if more etls are done
        help="Run a specific ETL pipeline: 'csu' or 'drough'"
    )

    parser.add_argument(
        '--check-dbs',
        action='store_true',
        help="Check the status and count of all databases."
    )

    return parser.parse_args()

def main():
    args = parse_args()

    embedding_service = TextEmbeddingService(ip="localhost", port=8000)
    db_repository = DbRepositoryFactory.create_db_repository(args.vector_db, ip="0.0.0.0", port=8001)

    
    if args.run_etl == 'csu':
        run_etl_csu(embedding_service, db_repository)
        
    elif args.run_etl == 'drough':
        run_etl_drough(embedding_service, db_repository)
        
    elif args.check_dbs:
        check_databases()
        
    else:
        # Default behavior: run the RAG chat
        logger.info("No ETL or DB check specified. Running RAG chat mode by default.")
        # run_rag_chat(embedding_service)

if __name__ == '__main__':
    main()
