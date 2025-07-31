from abc import ABC, abstractmethod
import logging
import traceback
import pandas as pd
from utils.logging_config import get_logger

logger = get_logger(__name__)

class BaseEtl(ABC):
    def __init__(self, filepath: str):
        super().__init__()
        self.documents = []
        self.df = None
        self.filepath = filepath

    def extract(self) -> None:
        try:
            self.df = pd.read_csv(self.filepath)
        except FileNotFoundError:
            logger.exception(f"File was not found: {self.filepath}")
            self.df = None
        except Exception as e:
            logger.exception(f"Error occurred during extraction: {e}")
            self.df = None

    @abstractmethod
    def transform(self):
        """
        This method has to be implemented by subclass.
        """
        raise NotImplementedError("Method .transform() has to be implemented in subclass.")
    
    def load(self) -> None:
        # TODO: sending to database
        logger.info("Previewing transformed data (first 5 rows):")
        print(self.df.head(5).to_string(index=True))

    def run(self) -> None:
        self.extract()
        if self.df is None:
            logger.error("Extraction failed. Stopping ETL")
            return
        
        try:
            logging.debug(f"Transforming data for {self.filepath}")
            self.transform()
            logging.debug(f"Transformation for {self.filepath} complete")
        except Exception as e:
            logging.exception(f"Error during transformation: {e}")
            traceback.print_exc()
            return

        try:
            logging.debug("Loading data...")
            self.load()
            logging.debug("Loading data done")
        except Exception as e:
            logging.exception(f"Error during loading: {e}")
            traceback.print_exc()