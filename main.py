import argparse
import time
from dotenv import load_dotenv
from typing import Dict
import datetime

from database.ChromaDbRepository import ChromaDbRepository
from database.QdrantDbRepository import QdrantDbRepository
from database.WeaviateDbRepository import WeaviateDbRepository
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
from database.base.MyDocument import MyDocument

setup_logging()
logger = get_logger(__name__)

load_dotenv()

def run_etl_csu(embedding_service: TextEmbeddingService, db_repositories: Dict[str, BaseDbRepository]):
    """Runs the CSU (Mzdr1DataEtl) ETL pipeline."""
    # TODO: Update this whole function
    highlight_log(logger, "Starting ETL pipeline CSU")

    for _, repository in db_repositories:
        repository.collection_name = constants.COLLECTION_NAME_CSU
        repository.metadata=constants.CHROMA_DB_METADATA_CSU
    
    etl = Mzdr1DataEtl(
        filepath="data/csu/MZDR_1_data.csv", 
        db_repository=db_repositories,
        embedding_service=embedding_service
    )
    etl.run()
    highlight_log(logger, "CSU ETL pipeline finished.")

def run_etl_drough(embedding_service: TextEmbeddingService, db_repositories: Dict[str, BaseDbRepository], llm_handler: LLMHandler):
    """Runs the Drough (PDF) ETL pipeline on all files."""
    highlight_log(logger, "Starting Drough ETL pipeline...")
    pdf_files = Utils.find_files(folder_path='/Users/david/Mendelu/Diplomka/drough_data/files', file_type='pdf')

    # setting db constants
    for _, repository in db_repositories.items():
        repository.collection_name = constants.COLLECTION_NAME_DROUGH

        if repository.name == ChromaDbRepository.name:
            repository.metadata = constants.CHROMA_DB_METADATA_DROUGH
        repository.connect_and_create_collection()

    # metadata extractor
    metadata_extractor = LLMMetadataExtractor(llm_handler=llm_handler)

    for file in pdf_files:
        start_time = time.time()

        obj = DroughEtl(
                filepath=file,
                db_repositories=db_repositories,
                embedding_service=embedding_service,
                metadata_extractor=metadata_extractor
            )
        status = obj.run()

        end_time = time.time()
        elapsed = end_time - start_time
        delta = datetime.timedelta(seconds=elapsed)
        highlight_log(logger, str(delta), character='~')
        
        if not status:
            logger.warning("ETL failed, stopping entire pipeline...")
            break
    highlight_log(logger, "Drough ETL pipeline finished.")

def check_databases(db_repositories: Dict[str, BaseDbRepository], collection_name: str):
    """Checks the record counts for configured databases."""
    highlight_log(logger, "Checking database counts...")
    for _, db in db_repositories.items():
        db.collection_name = collection_name

        if db.name == ChromaDbRepository.name:
            db.metadata = constants.CHROMA_DB_METADATA_DROUGH

        db.connect()
        db.logger.info(f"Rows in {collection_name}: {db.get_count()}")
        db.close()
    

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
    start_time = time.time()
    args = parse_args()

    # Objects
    embedding_service = TextEmbeddingService(ip="localhost", port=8000)
    # DbRepositoryFactory.create_all_repositories()
    db_repositories = DbRepositoryFactory.create_all_repositories()
    llm_handler = LLMHandler(ip="localhost", port=1234)

    if args.test_search:    
        # db_repository.collection_name = constants.COLLECTION_NAME_DROUGH
        # db_repository.metadata = constants.CHROMA_DB_METADATA_DROUGH
        # db_repository.connect()
        # db_repository.create_collection()

        # search_str = "What was the 2022 carbon dioxide concentration of 417.1 ppm?"
        # search_str_embed = embedding_service.get_embedding_with_uuid(data=search_str)
        # results = db_repository.search(text=None, text_embedded=search_str_embed[0].embedding, n_results=10)
        # if results.success:
        #     for d in results.data:
        #         print(d.text)
        #         print()
        # else:
        #     print('lol sussy baka')
        pass
    else:
        if args.data == 'csu' and args.run_etl:
            run_etl_csu(embedding_service, db_repositories)
            
        elif args.data == 'drough' and args.run_etl:
            run_etl_drough(embedding_service, db_repositories, llm_handler)
            
        elif args.check_dbs and args.data is not None:
            check_databases(db_repositories, args.data)
            
        else:
            # Default behavior: run the RAG chat
            logger.info("No ETL or DB check specified. Running RAG chat mode by default.")
            run_rag_chat(embedding_service, args.data, args.model_path)

        # Close dbs
        for _, db in db_repositories.items():
            db.close()
        
        end_time = time.time()
        elapsed = end_time - start_time
        delta = datetime.timedelta(seconds=elapsed)
        highlight_log(logger, str(delta), character='~')

if __name__ == '__main__':
    main()


    