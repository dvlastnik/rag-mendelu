from abc import ABC, abstractmethod
from typing import List
import pandas as pd

from typing import Type
from database.base.BaseDbRepository import BaseDbRepository
from etl.EtlState import ETLState
from utils.logging_config import get_logger
from text_embedding_api.TextEmbeddingService import TextEmbeddingService
from database.base.MyDocument import MyDocument
from utils.utils import Utils


logger = get_logger(__name__)

class BaseEtl(ABC):
    def __init__(self, filepath: str, db_repository: BaseDbRepository):
        super().__init__()
        self.documents = []
        self.df = None
        self.db_repository = db_repository
        self.filepath = filepath
        self.state = ETLState.NOT_STARTED

        # TODO: Make this somehow pretty, so we dont have to put hardcoded repositories here
        self.db_repository.connect()

    @abstractmethod
    def _row_to_document(self, row) -> MyDocument:
        raise NotImplementedError("Method ._row_to_document() has to be implemented in subclass.")

    def extract(self) -> None:
        try:
            self.df = pd.read_csv(self.filepath)
            self.state = ETLState.EXTRACTED
        except FileNotFoundError:
            self.df = None
            self.state = ETLState.FAILED
        except Exception as e:
            self.df = None
            self.state = ETLState.FILE_NOT_FOUND

    @abstractmethod
    def transform(self):
        """
        This method has to be implemented by subclass.
        """
        raise NotImplementedError("Method .transform() has to be implemented in subclass.")
    
    def load(self, embedding_service: TextEmbeddingService) -> None:
        documents: List[MyDocument] = self.df.apply(self._row_to_document, axis=1).tolist()
        texts = [doc.text for doc in documents]

        embeddings_response = embedding_service.get_embedding_with_uuid(texts, chunk_size=200)
        for doc, embed_text in zip(documents, embeddings_response):
            doc.embedding = embed_text.embedding
            doc.id = embed_text.uuid
        
        for i, doc in enumerate(Utils.chunks(documents, 500)):
            logger.info(f"{i}. chunk inserted")
            result = self.db_repository.insert(doc)
            if not result.success:
                logger.error(result.message)
                self.state = ETLState.FAILED
                return

        check_db_data = self.db_repository.check_if_data_were_inserted()
        if not check_db_data.success:
            logger.error(result.message)
            self.state = ETLState.FAILED
            return
        self.state = ETLState.LOADED

    def run(self, embedding_service: TextEmbeddingService) -> None:
        while True:
            match self.state:
                case ETLState.NOT_STARTED:
                    self.extract()
                case ETLState.EXTRACTED:
                    logger.info("Extraction done")
                    self.transform()
                case ETLState.TRANSFORMED:
                    logger.info("Transformation done")
                    self.load(embedding_service)
                case ETLState.LOADED:
                    logger.info("Loading done")
                    break
                case ETLState.FILE_NOT_FOUND:
                    logger.error(f"File was not found for path: {self.filepath}")
                    break
                case ETLState.FAILED:
                    logger.error(f"ETL failed")
                    break
                case _:
                    logger.warning("State machine in default case")
                    break