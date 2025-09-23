from abc import ABC, abstractmethod
from typing import List, Dict, Type

from database.base.MyDocument import MyDocument
from database.base.DbOperationResult import DbOperationResult
from utils.logging_config import get_logger

logger = get_logger(__name__)

class BaseDbRepository(ABC):
    def __init__(self, ip: str, port: int, collection_name: str, metadata: Dict):
        self.ip = ip
        self.port = port
        self.collection_name = collection_name
        self.metadata = metadata
        super().__init__()

    @abstractmethod
    def connect(self, create_collection: bool) -> DbOperationResult:
        pass

    @abstractmethod
    def search(self, text: str | List[str] | None, text_embedded: List[int] | List[List[int]] | None=None, n_results: int=3) -> DbOperationResult:
        pass

    @abstractmethod
    def check_if_data_were_inserted(self) -> DbOperationResult:
        pass

    @abstractmethod
    def get_count(self) -> int:
        pass

    @abstractmethod
    def insert(self, docs: List[MyDocument]) -> DbOperationResult:
        pass

    @abstractmethod
    def delete(self, ids: List[str]) -> DbOperationResult:
        pass

    @abstractmethod
    def close(self) -> DbOperationResult:
        pass

    @classmethod
    def check_count_for(cls, repo_cls: Type["BaseDbRepository"], ip: str, port: int, collection_name: str, metadata: Dict) -> None:
        repository = repo_cls(ip=ip, port=port, collection_name=collection_name, metadata=metadata)

        res = repository.connect(create_collection=False)
        if res.success:
            count = repository.get_count()
            logger.info(f"{repo_cls.__name__} has {count} rows")
            repository.close()
        else:
            logger.warning(f"{repo_cls.__name__} did not connect to DB :[")



