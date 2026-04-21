from typing import Protocol, List, Dict, Any, Optional
from dataclasses import dataclass, field

@dataclass
class ScientificSearchResult:
    text: str
    score: float
    source_id: str
    variable: str
    metadata: Dict[str, Any] = field(default_factory=dict)

class ScientificDataSource(Protocol):
    def search(
        self,
        query: str,
        top_k: int = 10,
        source_id: Optional[str] = None,
        variable: Optional[str] = None,
    ) -> List[ScientificSearchResult]: 
        ...

    def is_available(self) -> bool:
        ...