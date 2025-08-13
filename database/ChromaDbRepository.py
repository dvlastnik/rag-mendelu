from typing import List
import chromadb
from chromadb.config import Settings

from database.base.Document import Document
from database.base.BaseDbRepository import BaseDbRepository
from database.base.DbOperationResult import DbOperationResult
from utils.logging_config import get_logger

logger = get_logger(__name__)

class ChromaDbRepository(BaseDbRepository):
    COLLECTION_NAME = 'csu'
    METADATA = {
        "name": "MZDCRR_1_data",
        "description": "Průměrný evidenční počet zaměstnanců a průměrné hrubé měsíční mzdy z čtvrtletního zjišťování - kumulace čtvrtletí za 1-4Q (rok).",
        "source": "Český statistický úřad (ČSÚ)"
    }

    def _if_collection_exist_delete(self, collection_name: str):
        exists = self.get_collection(collection_name=collection_name)
        if exists:
            self.client.delete_collection(collection_name)

    def _create_collection(self, collection_name: str, metadata: dict) -> chromadb.Collection:
        try:
            self._if_collection_exist_delete(collection_name=collection_name)
        except chromadb.errors.NotFoundError:
            logger.debug(f"Collection {self.COLLECTION_NAME} was not yet created")

        return self.client.create_collection(
            name=collection_name,
            metadata=metadata
        )

    def get_collection(self, collection_name: str):
        self.collection = self.client.get_collection(
            name=collection_name
        )

        return self.collection

    def connect(self, create_collection=True):
        self.client = chromadb.HttpClient(host=self.ip, port=self.port, settings=Settings(anonymized_telemetry=False))
        if create_collection:
            self.collection = self._create_collection(collection_name=self.COLLECTION_NAME, metadata=self.METADATA)
        else:
            self.collection = self.get_collection(self.COLLECTION_NAME)

    def close(self):
        return super().close()

    def insert(self, docs: List[Document]):
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

            results = Document.from_chromadb_result(results)
            return DbOperationResult(success=True, data=results)
        except Exception as e:
            return DbOperationResult(success=False, message=f"Error during search at ChromaDbRepository: {e}")

    def check_if_data_were_inserted(self):
        data = self.collection.get()

        if len(data["ids"]) > 0:
            return DbOperationResult(success=True)
        return DbOperationResult(success=False, message="When trying to get all data from database empty list was returned")
        