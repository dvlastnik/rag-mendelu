from dataclasses import dataclass, field
from typing import List


@dataclass
class SparseVectorData:
    indices: List[int]
    values: List[float]


@dataclass
class EmbeddingResponse:
    uuid: str
    embedding: List[float]
    sparse: SparseVectorData | None = field(default=None)

    def __str__(self):
        return f"Uuid: {self.uuid}, Embedding: {self.embedding}, Sparse: {self.sparse}"
