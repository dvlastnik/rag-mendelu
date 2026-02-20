from abc import ABC, abstractmethod
import pathlib
from typing import List, Dict
import pandas as pd
import traceback

from typing import Type
from database.base.BaseDbRepository import BaseDbRepository
from etl.EtlState import ETLState
from utils.Utils import Utils
from utils.logging_config import get_logger, highlight_log
from etl.converters import convert_data
from text_embedding import TextEmbeddingService
from database.base.MyDocument import MyDocument


logger = get_logger(__name__)

class BaseEtl(ABC):
    OUTPUT_FOLDER: str = "data"

    def __init__(self, filepath: str, db_repositories: Dict[str, BaseDbRepository], embedding_service: TextEmbeddingService):
        super().__init__()
        self.documents: List[MyDocument] = []
        # dataframe for csv and xlsx
        self.df = None
        self.db_repositories = db_repositories
        self.embedding_service = embedding_service
        self.file = pathlib.Path(filepath)

        self.state = ETLState.NOT_STARTED

    def _insert_by_chunks(self, chunk_size: int = 500) -> ETLState:
        logger.info(f"Inserting documents of lenght: {len(self.documents)}, chunk_size = {chunk_size}")
        for i, doc in enumerate(Utils.chunks(self.documents, chunk_size)):
            for _, repository in self.db_repositories.items():
                result = repository.insert(doc)
                if not result.success:
                    logger.error(result.message)
                    return ETLState.FAILED
                else:
                    index = i+1
                    logger.info(f"{index}. chunk inserted ({repository.name})")

        return ETLState.LOADED

    def _check_if_data_are_loaded(self) -> ETLState:
        for _, repository in self.db_repositories.items():
            check_db_data = repository.check_if_data_were_inserted()
            if not check_db_data.success:
                logger.error(check_db_data.message)
                return ETLState.FAILED

            logger.info(f"Rows in database after loading: {self.file.stem} is {repository.get_count()}")
        return ETLState.LOADED

    @abstractmethod
    def get_file_path(self, only_folder: bool = False, chunk_index: int | None = None) -> pathlib.Path:
        raise NotImplementedError("Method .get_file_path() has to be implemented in subclass.")

    def extract(self) -> None:
        try:
            try:
                self.df = convert_data(self.file, output_folder=self.OUTPUT_FOLDER)
                self.state = ETLState.EXTRACTED
            except ValueError as ve:
                logger.error(ve)
                self.state = ETLState.FAILED
        except FileNotFoundError:
            self.df = None
            self.state = ETLState.FILE_NOT_FOUND
        except Exception as e:
            self.df = None
            traceback.print_exc()
            self.state = ETLState.FAILED

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
            if self.state is ETLState.FAILED:
                return
            self.state = self._check_if_data_are_loaded()
            if self.state is ETLState.FAILED:
                return
            self.state = ETLState.LOADED
        except Exception as e:
            logger.exception(f"ETL Load strategy '{self.file.suffix.lower()}' failed: {e}")
            self.state = ETLState.FAILED

    def run(self) -> bool:
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
                    return True
                case ETLState.FILE_NOT_FOUND:
                    logger.error(f"File was not found for path: {self.file}")
                    return False
                case ETLState.FAILED:
                    logger.error(f"ETL failed")
                    return False
                case _:
                    logger.warning("State machine in default case")
                    return False