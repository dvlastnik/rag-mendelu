from typing import List, Dict
from dataclasses import dataclass

@dataclass
class Document:
    id: str
    text: str
    embedding: List[float] = None
    metadata: Dict = None

    @staticmethod
    def from_chromadb_result(data: Dict) -> 'Document' | List['Document']:
        length = len(data['ids'])

        if length > 1:
            result_list = []
            for i in range(length):
                result_list.append(Document(
                    id=data['ids'][i],
                    text=data['documents'][i],
                    metadata=data['metadatas'][i]
                ))

            return result_list
        return Document(id=data['ids'][0], text=data['documents'][0], metadata=data['metadatas'][0])

        