from abc import ABC, abstractmethod
from typing import List

from database.base.Document import Document
from database.base.DbOperationResult import DbOperationResult

class BaseDbRepository(ABC):
    def __init__(self, ip: str, port: int):
        self.ip = ip
        self.port = port
        super().__init__()

    @abstractmethod
    def connect(self, create_collection: bool) -> DbOperationResult:
        pass

    @abstractmethod
    def search(self, text: str | List[str], text_embedded: List[int] | List[List[int]] | None=None, n_results: int=3) -> DbOperationResult:
        pass

    @abstractmethod
    def check_if_data_were_inserted(self) -> DbOperationResult:
        pass

    @abstractmethod
    def insert(self, docs: List[Document]) -> DbOperationResult:
        pass

    @abstractmethod
    def delete(self, ids: List[str]) -> DbOperationResult:
        pass

    @abstractmethod
    def close(self) -> DbOperationResult:
        pass