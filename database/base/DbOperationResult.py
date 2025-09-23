from dataclasses import dataclass
from typing import Optional, List

from database.base.MyDocument import MyDocument

@dataclass
class DbOperationResult:
    success: bool
    message: Optional[str] = None
    data: Optional[List[MyDocument]] = None
