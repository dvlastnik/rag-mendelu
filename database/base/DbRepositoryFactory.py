from database.ChromaDbRepository import ChromaDbRepository
from database.base.BaseDbRepository import BaseDbRepository


class DbRepositoryFactory:
    @staticmethod
    def create_db_repository(type: str, ip: str, port: int) -> BaseDbRepository:
        if type == ChromaDbRepository.name:
            return ChromaDbRepository(ip=ip, port=port)
        
        raise Exception(f"{type} is not known. Maybe it exists, but it is not added to factory method in BaseDbRepository.")