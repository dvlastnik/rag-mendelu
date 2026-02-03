from typing import List, Dict, Optional, Any
from dataclasses import dataclass

@dataclass
class SparseVector:
    indices: List[int]
    values: List[float]

@dataclass
class MyDocument():
    id: str
    text: str
    embedding: List[float] = None
    sparse_embedding: SparseVector = None
    metadata: Dict = None
    score: Optional[float] = None

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'MyDocument':
        return MyDocument(
            id=str(data.get("id")),
            text=data.get("text", ""),
            metadata=data.get("meta") or data.get("metadata") or {},
            score=data.get("score"),
            embedding=data.get("embedding"),
            sparse_embedding=data.get("sparse_embedding")
        )

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

        