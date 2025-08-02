from abc import ABC, abstractmethod
import logging
import traceback
import pandas as pd
from utils.logging_config import get_logger

logger = get_logger(__name__)

from enum import Enum

class ETLState(Enum):
    NOT_STARTED = 1
    EXTRACTED = 2
    TRANSFORMED = 3
    LOADED = 4
    FAILED = 5
    FILE_NOT_FOUND = 6

class BaseEtl(ABC):
    def __init__(self, filepath: str):
        super().__init__()
        self.documents = []
        self.df = None
        self.filepath = filepath
        self.state = ETLState.NOT_STARTED

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
    
    def load(self) -> None:
        # TODO: sending to database
        total_rows = len(self.df)

        print("\n🟩 First 5 rows:")
        print(self.df.head(5).to_string(index=True))

        if total_rows > 10:
            middle_start = total_rows // 2 - 2
            middle_end = middle_start + 5
            print("\n🟨 Middle 5 rows:")
            print(self.df.iloc[middle_start:middle_end].to_string(index=True))
        else:
            print("\n🟨 Not enough rows for middle preview.")

        print("\n🟥 Last 5 rows:")
        print(self.df.tail(5).to_string(index=True))
        self.state = ETLState.LOADED

    def run(self) -> None:
        while True:
            match self.state:
                case ETLState.NOT_STARTED:
                    self.extract()
                case ETLState.EXTRACTED:
                    logger.info("Extraction done")
                    self.transform()
                case ETLState.TRANSFORMED:
                    logger.info("Transformation done")
                    self.load()
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