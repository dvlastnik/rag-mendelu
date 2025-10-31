from abc import ABC, abstractmethod
import pathlib
from typing import List
import pandas as pd

from typing import Type
from database.base.BaseDbRepository import BaseDbRepository
from etl.EtlState import ETLState
from utils.Utils import Utils
from utils.logging_config import get_logger, highlight_log
from etl.converters import convert_data
from text_embedding_api.TextEmbeddingService import TextEmbeddingService
from database.base.MyDocument import MyDocument


logger = get_logger(__name__)

class BaseEtl(ABC):
    def __init__(self, filepath: str, db_repository: BaseDbRepository, embedding_service: TextEmbeddingService):
        super().__init__()
        self.documents: List[MyDocument] = []
        # dataframe for csv and xlsx
        self.df = None
        self.db_repository = db_repository
        self.embedding_service = embedding_service
        self.file = pathlib.Path(filepath)

        self.state = ETLState.NOT_STARTED

    def _insert_by_chunks(self, chunk_size: int = 500) -> ETLState:
        logger.info(f"Inserting documents of lenght: {len(self.documents)}, chunk_size = {chunk_size}")
        for i, doc in enumerate(Utils.chunks(self.documents, chunk_size)):
            result = self.db_repository.insert(doc)
            if not result.success:
                logger.error(result.message)
                return ETLState.FAILED
            else:
                logger.info(f"{i}. chunk inserted")

        return ETLState.LOADED

    def _check_if_data_are_loaded(self) -> ETLState:
        check_db_data = self.db_repository.check_if_data_were_inserted()
        if not check_db_data.success:
            logger.error(check_db_data.message)
            return ETLState.FAILED

        logger.info(f"Rows in database after loading: {self.file.stem} is {self.db_repository.get_count()}")
        return ETLState.LOADED

    @abstractmethod
    def _row_to_document(self, row) -> MyDocument:
        raise NotImplementedError("Method ._row_to_document() has to be implemented in subclass.")
    
    @abstractmethod
    def get_file_path(self, only_folder: bool = False, chunk_index: int | None = None) -> pathlib.Path:
        raise NotImplementedError("Method .get_file_path() has to be implemented in subclass.")

    def extract(self) -> None:
        try:
            try:
                self.df = convert_data(self.file)
                self.state = ETLState.EXTRACTED
            except ValueError as ve:
                logger.error(ve)
                self.state = ETLState.FAILED
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
    
    def load(self) -> None:
        from etl.loaders import load_data

        try:
            self.state = load_data(self)
            self.state = self._check_if_data_are_loaded()
        except Exception as e:
            logger.exception(f"ETL Load strategy '{self.file.suffix.lower()}' failed: {e}")
            self.state = ETLState.FAILED

    def run(self) -> None:
        logger.info(50*'=')
        while True:
            match self.state:
                case ETLState.NOT_STARTED:
                    highlight_log(logger=logger, text="Extraction", character='*', only_char=False)
                    self.extract()
                case ETLState.EXTRACTED:
                    highlight_log(logger=logger, text="Extraction", character='*', only_char=True)
                    highlight_log(logger=logger, text="Transformation", character='*', only_char=False)
                    self.transform()
                case ETLState.TRANSFORMED:
                    highlight_log(logger=logger, text="Transformation", character='*', only_char=True)
                    highlight_log(logger=logger, text="Loading", character='*', only_char=False)
                    self.load()
                case ETLState.LOADED:
                    highlight_log(logger=logger, text="Loading", character='*', only_char=True)
                    logger.info(50*'=')
                    print()
                    break
                case ETLState.FILE_NOT_FOUND:
                    logger.error(f"File was not found for path: {self.file}")
                    break
                case ETLState.FAILED:
                    logger.error(f"ETL failed")
                    break
                case _:
                    logger.warning("State machine in default case")
                    break