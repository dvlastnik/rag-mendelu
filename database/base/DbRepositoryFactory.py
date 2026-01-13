import os
from typing import Dict, List
from database.base.BaseDbRepository import BaseDbRepository

# Import all available repository implementations
from database.ChromaDbRepository import ChromaDbRepository
from database.QdrantDbRepository import QdrantDbRepository
from utils.logging_config import get_logger

logger = get_logger(__name__)

class DbRepositoryFactory:
    
    @staticmethod
    def create_db_repository(type: str, 
                             ip: str, 
                             port: int) -> BaseDbRepository:
        """
        Creates a single instance of a specified repository.
        """
        logger.info(f"Factory creating single instance of type: {type}")
        
        if type == ChromaDbRepository.name:
            return ChromaDbRepository(ip=ip, port=port)
        elif type == QdrantDbRepository.name:
            return QdrantDbRepository(ip=ip, port=port)
        
        raise Exception(f"{type} is not known. Maybe it exists, but it is not added to factory method in BaseDbRepository.")

    @staticmethod
    def create_all_repositories(ip: str = "localhost") -> Dict[str, BaseDbRepository]:
        """
        Creates and returns a dictionary of all available DbRepository instances,
        keyed by their static 'name' attribute.
        
        Make sure load_dotenv() has been called before running this.
        """
        logger.debug(f"Factory creating all repositories")
        repositories: Dict[str, BaseDbRepository] = {}
        
        try:
            chroma_port = int(os.environ.get("CHROMA_PORT", 8001))
            logger.info(f"Initializing ChromaDbRepository (Host: {ip}, Port: {chroma_port})")
            repositories[ChromaDbRepository.name] = ChromaDbRepository(
                ip=ip,
                port=chroma_port
            )
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDbRepository: {e}", exc_info=True)

        db_metadata = {
            "vector_size": int(os.environ.get("VECTOR_DB_VECTOR_SIZE", 384)),
            "distance": str(os.environ.get("VECTOR_DB_DISTANCE", "DOT"))
        }

        try:
            qdrant_grpc_port = int(os.environ.get("QDRANT_GRPC_PORT"))
            qdrant_http_port = int(os.environ.get("QDRANT_REST_PORT"))
            logger.debug(f"Initializing QdrantDbRepository (Host: {ip}, Port: {qdrant_http_port}, Grpc port: {qdrant_grpc_port})")
            repositories[QdrantDbRepository.name] = QdrantDbRepository(
                ip=ip,
                port=qdrant_http_port,
                grpc_port=qdrant_grpc_port,
                collection_name="drough",
                metadata=db_metadata
            )
        except Exception as e:
            logger.error(f"Failed to initialize QdrantDbRepository: {e}", exc_info=True)

        # try:
        #     weaviate_port = int(os.environ.get("WEAVIATE_HTTP_PORT", 8080))
        #     weaviate_grpc_port = int(os.environ.get("WEAVIATE_GRPC_PORT", 50051))
        #     logger.debug(f"Initializing WeaviateDbRepository (Host: {ip}, Port: {weaviate_port})")
        #     repositories[WeaviateDbRepository.name] = WeaviateDbRepository(
        #         ip=ip,
        #         port=weaviate_port,
        #         grpc_port=weaviate_grpc_port,
        #         metadata=db_metadata
        #     )
        # except Exception as e:
        #     logger.error(f"Failed to initialize WeaviateDbRepository: {e}", exc_info=True)

        logger.debug(f"Factory created {len(repositories)} repository instances: {list(repositories.keys())}")
        return repositories