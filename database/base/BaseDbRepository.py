from abc import ABC, abstractmethod
from typing import List, Dict, Type, Any

from database.base.MyDocument import MyDocument
from database.base.DbOperationResult import DbOperationResult, execute_and_check_db_operation
from utils.logging_config import get_logger

logger = get_logger(__name__)

class BaseDbRepository(ABC):
    name = 'base'

    def __init__(self, ip: str, port: int, collection_name: str = "", metadata: Dict = {}):
        self.ip = ip
        self.port = port
        self.collection_name = collection_name
        self.metadata = metadata
        self.logger = get_logger(self.name)

    def connect_and_create_collection(self):
        execute_and_check_db_operation(operation=self.connect, operation_description=f".connect() {self.name}")
        execute_and_check_db_operation(operation=self.if_collection_exist_delete, operation_description=f"if_collection_exist_delete {self.name}")
        self.create_collection()

    @abstractmethod
    def connect(self) -> DbOperationResult:
        pass

    @abstractmethod
    def create_collection(self) -> Any:
        pass

    @abstractmethod
    def search(self, text: str | List[str] | None = None, text_embedded: List[int] | List[List[int]] | None=None, n_results: int=3) -> DbOperationResult:
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

    @abstractmethod
    def if_collection_exist_delete(self) -> DbOperationResult:
        pass

    @classmethod
    def check_count_for(cls, repo_cls: Type["BaseDbRepository"], ip: str, port: int, collection_name: str, metadata: Dict = {}) -> None:
        repository = repo_cls(ip=ip, port=port, collection_name=collection_name, metadata=metadata)

        res = repository.connect()
        if res.success:
            count = repository.get_count()
            logger.info(f"{repo_cls.__name__} has {count} rows")
            repository.close()
        else:
            logger.warning(f"{repo_cls.__name__} did not connect to DB :[")



