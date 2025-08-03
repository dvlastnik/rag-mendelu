from dataclasses import dataclass
from typing import Optional, List

from database.base.Document import Document

@dataclass
class DbOperationResult:
    success: bool
    message: Optional[str] = None
    data: Optional[List[Document]]
