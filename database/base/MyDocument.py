from typing import List, Dict
from dataclasses import dataclass

@dataclass
class MyDocument():
    id: str
    text: str
    embedding: List[float] = None
    metadata: Dict = None

    @staticmethod
    def from_chromadb_result(data: Dict) -> 'MyDocument' | List['MyDocument']:
        ids = data['ids'][0] if isinstance(data['ids'][0], list) else data['ids']
        documents = data['documents'][0] if isinstance(data['documents'][0], list) else data['documents']
        metadatas = data['metadatas'][0] if isinstance(data['metadatas'][0], list) else data['metadatas']

        length = len(ids)
        if length > 1:
            return [MyDocument(
                id=ids[i],
                text=documents[i],
                metadata=metadatas[i]
            ) for i in range(length)]
        
        return MyDocument(id=ids[0], text=documents[0], metadata=metadatas[0])

        