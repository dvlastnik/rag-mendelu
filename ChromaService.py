import chromadb

from ChromaDbModel import ChromaDbModel

class ChromaDbService():
    MAX_BATCH_SIZE = 500

    def __init__(self, url: str):
        if ":" not in url:
            raise ValueError(f"Not a valid url address: {url}")
        url = url.split(sep=':')
        ip = url[0]
        port = url[1]

        self.client = chromadb.HttpClient(host=ip, port=port)
        self.collection = None

    def _if_collection_exist_delete(self, collection_name: str):
        exists = self.get_collection(collection_name=collection_name)
        if exists:
            self.delete_collection(collection_name)

    def create_collection(self, collection_name: str, metadata: dict):
        self._if_collection_exist_delete(collection_name=collection_name)

        self.collection = self.client.create_collection(
            name=collection_name,
            metadata=metadata
        )

    def get_collection(self, collection_name: str):
        self.collection = self.client.get_collection(
            name=collection_name
        )

        return self.collection
    
    def add_data(self, data: list[ChromaDbModel]):
        ids = [model.id for model in data]
        embeddings = [model.text_embedding for model in data]
        documents = [model.text for model in data]
        metadatas = [model.metadata for model in data]

        for i in range(0, len(ids), self.MAX_BATCH_SIZE):
            batch_ids = ids[i:i + self.MAX_BATCH_SIZE]
            batch_embeddings = embeddings[i:i + self.MAX_BATCH_SIZE]
            batch_documents = documents[i:i + self.MAX_BATCH_SIZE]
            batch_metadatas = metadatas[i:i + self.MAX_BATCH_SIZE]

            self.collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                documents=batch_documents,
                metadatas=batch_metadatas
            )

            print(f"Added batch {i // self.MAX_BATCH_SIZE + 1} of {len(ids) // self.MAX_BATCH_SIZE + 1}")

    def delete_collection(self, name: str):
        self.client.delete_collection(name=name)

    def query(self, query: list[int]):
        result = self.collection.query(
            query_embeddings=query,
            n_results=5
        )

        return result
    
    def from_query_result_to_context_str(self, result: chromadb.QueryResult) -> str:
        metadatas = result.get('metadatas')
        documents = result.get('documents')
        metadatas = metadatas[0]
        documents = documents[0]

        context = ''
        for i in range(len(documents)):
            temp_context = f"Document Source: {metadatas[i]['source']}\n"
            temp_context += f"Document Content:\n{documents[i]}\n"
            
            context += f"{temp_context}\n\n"
        return context