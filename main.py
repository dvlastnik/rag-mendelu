import sys
import os
from database.ChromaDbRepository import ChromaDbRepository
from database.base.BaseDbRepository import BaseDbRepository
from text_embedding_api.TextEmbeddingService import TextEmbeddingService
from etl.Mzdr1DataEtl import Mzdr1DataEtl
from rag.RAG import RAG
from utils.logging_config import get_logger, setup_logging
import constants

setup_logging()
logger = get_logger(__name__)

def main():
    print(f"CPU COUNT IS: {os.cpu_count()}")
    if len(sys.argv) >= 2:
        if sys.argv[1] == "--run-etl":
            etl = Mzdr1DataEtl(
                filepath="data/MZDR_1_data.csv", 
                db_repository=ChromaDbRepository(
                    ip="localhost",
                    port=8001,
                    collection_name=constants.COLLECTION_NAME,
                    metadata=constants.CHROMA_DB_METADATA
                )
            )
            embedding_service = TextEmbeddingService(ip="localhost", port=8000)
            etl.run(embedding_service)
        elif sys.argv[1] == "--check-dbs":
            # TODO: Create script that will accept some file, load it and will check all dbs that are in the config file
            # so it will not be hardcoded
            BaseDbRepository.check_count_for(ChromaDbRepository, ip="localhost", port=8001, collection_name=constants.COLLECTION_NAME, metadata=constants.CHROMA_DB_METADATA)
    else:
        embedding_service = TextEmbeddingService("localhost", 8000)
        question = "Průměrný evidenční počet zaměstnanců přepočtený (tis. osob) v roce 2000?"
        embed_text = embedding_service.get_embedding_with_uuid(question, chunk_size=None)
        
        chroma = ChromaDbRepository(ip="localhost", port=8001, collection_name=constants.COLLECTION_NAME, metadata=constants.CHROMA_DB_METADATA)
        chroma.connect(create_collection=False)
        
        rag = RAG(database_service=chroma, embedding_service=embedding_service, model_path="/Users/david/Library/Application Support/nomic.ai/GPT4All/Meta-Llama-3.1-8B-Instruct-128k-Q4_0.gguf")
        rag.chat("Jaký byl průměrný evidenční počet zaměstnanců přepočtený (tis. osob) v sektoru F - Stavebnictví v roce 2000?")

if __name__ == '__main__':
    main()
    

    

