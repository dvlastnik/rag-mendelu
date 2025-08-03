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
    def connect(self) -> DbOperationResult:
        pass

    @abstractmethod
    def search(self) -> DbOperationResult:
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