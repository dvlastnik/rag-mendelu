import argparse
import re
import lmstudio as lms

from database.ChromaDbRepository import ChromaDbRepository
from database.base.DbRepositoryFactory import DbRepositoryFactory
from database.base.BaseDbRepository import BaseDbRepository
from etl.DroughEtl import DroughEtl
from llm_handler.LLMHandler import LLMHandler
from metadata_extractor.LLMMetadataExtractor import LLMMetadataExtractor
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

def run_etl_drough(embedding_service: TextEmbeddingService, db_repository: BaseDbRepository, llm_handler: LLMHandler):
    """Runs the Drough (PDF) ETL pipeline on all files."""
    highlight_log(logger, "Starting Drough ETL pipeline...")
    pdf_files = Utils.find_files(folder_path='/Users/david/Mendelu/Diplomka/drough_data/files', file_type='pdf')

    # setting db constants
    db_repository.collection_name = constants.COLLECTION_NAME_DROUGH
    db_repository.metadata = constants.CHROMA_DB_METADATA_DROUGH
    db_repository.connect_and_create_collection()

    # metadata extractor
    metadata_extractor = LLMMetadataExtractor(llm_handler=llm_handler)

    for file in pdf_files:
        obj = DroughEtl(
                filepath=file,
                db_repository=db_repository,
                embedding_service=embedding_service,
                metadata_extractor=metadata_extractor
            )
        status = obj.run()
        if not status:
            logger.warning("ETL failed, stopping entire pipeline...")
            break
    highlight_log(logger, "Drough ETL pipeline finished.")

def check_databases():
    """Checks the record counts for configured databases."""
    highlight_log(logger, "Checking database counts...")
    # TODO: Create script that will accept some file, load it and will check all dbs that are in the config file
    # so it will not be hardcoded
    # BaseDbRepository.check_count_for(ChromaDbRepository, ip="localhost", port=8001, collection_name=constants.COLLECTION_NAME_CSU, metadata=constants.CHROMA_DB_METADATA_CSU)
    BaseDbRepository.check_count_for(ChromaDbRepository, ip="localhost", port=8001, collection_name=constants.COLLECTION_NAME_DROUGH, metadata=constants.CHROMA_DB_METADATA_DROUGH)


def run_rag_chat(embedding_service: TextEmbeddingService, data_name: str, model_path: str):
    """
    Starts the RAG chat mode.
    """
    highlight_log(logger, "Starting RAG chat mode...")

    collection_name = constants.COLLECTION_NAME_DROUGH
    collection_metadata = constants.CHROMA_DB_METADATA_DROUGH
    if data_name == 'csu':
        collection_name = constants.COLLECTION_NAME_CSU
        collection_metadata = constants.CHROMA_DB_METADATA_CSU

    chroma = ChromaDbRepository(ip="localhost", port=8001, collection_name=collection_name, metadata=collection_metadata)
    connect_result = chroma.connect()
    
    if not connect_result.success:
        logger.error(f"Failed to connect to ChromaDB for RAG: {connect_result.message}")
        return

    rag = RAG(database_service=chroma, embedding_service=embedding_service, model_path=model_path)

    # question = "What was the annual global average carbon dioxide concentration in 2022, and what percentage is that above the pre-industrial level?"
    question = "In 2022, the annual global average carbon dioxide concentration in the atmosphere rose to 417.1±0.1 ppm, which is 50% greater than the pre-industrial level. Global mean tropospheric methane abundance was 16% higher than its pre-industrial level, and nitrous oxide was 24% higher. All three gases set new record-high atmospheric concentration levels in 2022."
    result = rag.chat(question)
    print()
    logger.info("--- RAG Result ---")
    logger.info(f"Question: {question}")
    logger.info(f"Answer: {result['answer']}")
    for index, source in enumerate(result['sources']):
        logger.info(f"Source {index}:")
        logger.info(f"  Page_content: {source.page_content}")
        logger.info(f"  Metadata: {source.metadata}")

def parse_args():
    parser = argparse.ArgumentParser(description="Main app")

    parser.add_argument(
        '--test-search',
        action='store_true',
        help="Run a test"
    )

    parser.add_argument(
        '--vector-db',
        type=str,
        default='chroma',
        choices=['chroma', 'qdrant'], # TODO: Add other
        help="Type of vector database to use"
    )

    parser.add_argument(
        '--data',
        type=str,
        default='drough',
        choices=['csu', 'drough'],
        help="Dataset"
    )

    parser.add_argument(
        '--run-etl',
        action='store_true',
        help="Run a ETL"
    )

    parser.add_argument(
        '--check-dbs',
        action='store_true',
        help="Check the status and count of all databases."
    )

    return parser.parse_args()

def main():
    args = parse_args()

    # Objects
    embedding_service = TextEmbeddingService(ip="localhost", port=8000)
    db_repository = DbRepositoryFactory.create_db_repository(args.vector_db, ip="0.0.0.0", port=8001)
    llm_handler = LLMHandler(ip="localhost", port=1234)

    if args.test_search:    
        db_repository.collection_name = constants.COLLECTION_NAME_DROUGH
        db_repository.metadata = constants.CHROMA_DB_METADATA_DROUGH
        db_repository.connect()
        db_repository.create_collection()

        search_str = "What was the 2022 carbon dioxide concentration of 417.1 ppm?"
        search_str_embed = embedding_service.get_embedding_with_uuid(data=search_str)
        results = db_repository.search(text=None, text_embedded=search_str_embed[0].embedding, n_results=10)
        if results.success:
            for d in results.data:
                print(d.text)
                print()
        else:
            print('lol sussy baka')
    else:
        if args.data == 'csu' and args.run_etl:
            run_etl_csu(embedding_service, db_repository)
            
        elif args.data == 'drough' and args.run_etl:
            run_etl_drough(embedding_service, db_repository, llm_handler)
            
        elif args.check_dbs:
            check_databases()
            
        else:
            # Default behavior: run the RAG chat
            logger.info("No ETL or DB check specified. Running RAG chat mode by default.")
            run_rag_chat(embedding_service, args.data, args.model_path)

if __name__ == '__main__':
    main()