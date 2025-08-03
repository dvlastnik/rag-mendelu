from typing import List, Dict
from dataclasses import dataclass

@dataclass
class Document:
    id: str
    text: str
    embedding: List[float] = None
    metadata: Dict = None