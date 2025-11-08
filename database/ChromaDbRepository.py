from typing import List
import chromadb
from chromadb.config import Settings

from database.base.MyDocument import MyDocument
from database.base.BaseDbRepository import BaseDbRepository
from database.base.DbOperationResult import DbOperationResult
from utils.logging_config import get_logger

logger = get_logger(__name__)

class ChromaDbRepository(BaseDbRepository):
    name = 'chroma'
    
    def if_collection_exist_delete(self):
        try:
            exists = self.get_collection()
            if exists:
                logger.info(f"Collection '{self.collection_name}' exists, deleting.")
                self.client.delete_collection(self.collection_name)
                return DbOperationResult(success=True)
            else:
                return DbOperationResult(success=True)
        except chromadb.errors.NotFoundError:
            logger.debug(f"Collection {self.collection_name} already exists.")
            return DbOperationResult(success=True)

    def create_collection(self) -> chromadb.Collection:
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata=self.metadata,
            get_or_create=True
        )

        return self.collection

    def get_collection(self):
        self.collection = self.client.get_collection(
            name=self.collection_name
        )

        return self.collection

    def connect(self):
        try:
            self.client = chromadb.HttpClient(host=self.ip, port=self.port, settings=Settings(anonymized_telemetry=False))
            self.collection = self.create_collection()
            return DbOperationResult(success=True)
        except Exception as e:
            return DbOperationResult(success=False, message=f"Error occured during 'connect' function: {e}")


    def close(self):
        return super().close()

    def insert(self, docs: List[MyDocument]):
        ids = []
        embeddings = []
        texts = []
        metadatas = []
        
        for doc in docs:
            ids.append(doc.id)
            embeddings.append(doc.embedding)
            texts.append(doc.text)
            metadatas.append(doc.metadata)

        try:
            self.collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas
                )
            
            return DbOperationResult(success=True)
        except Exception as e:
            return DbOperationResult(success=False, message=f"Exception occured inside insert function: {e}")
        
    def delete(self, ids):
        return super().delete(ids)
    
    def search(self, text, text_embedded, n_results=3):
        try:
            results = None

            if text_embedded is not None:
                results = self.collection.query(
                    query_embeddings=text_embedded,
                    n_results=n_results
                )

            else:
                results = self.collection.query(
                    query_texts=text,
                    n_results=n_results
                )

            results = MyDocument.from_chromadb_result(results)
            return DbOperationResult(success=True, data=results)
        except Exception as e:
            return DbOperationResult(success=False, message=f"Error during search at ChromaDbRepository: {e}")

    def check_if_data_were_inserted(self):
        if self.get_count() > 0:
            return DbOperationResult(success=True)
        return DbOperationResult(success=False, message="When trying to get all data from database empty list was returned")
    
    def get_count(self):
        return self.get_collection().count()
        