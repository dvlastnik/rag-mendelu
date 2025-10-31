from dataclasses import dataclass
from typing import Optional, List, Callable, Any

from database.base.MyDocument import MyDocument
from utils.logging_config import get_logger

logger = get_logger(__name__)

@dataclass
class DbOperationResult:
    success: bool
    message: Optional[str] = None
    data: Optional[List[MyDocument]] = None

def execute_and_check_db_operation(
    operation: Callable[..., DbOperationResult],
    operation_description: str,
    *args: Any,
    **kwargs: Any
) -> DbOperationResult:
    """
    Executes a given function and immediately checks its DbOperationResult.

    Args:
        operation: The function to execute (e.g., self.db_repository.connect).
        operation_description: A string to identify the operation for error messages.
        *args: Positional arguments to pass to the operation.
        **kwargs: Keyword arguments to pass to the operation.

    Returns:
        The DbOperationResult object if the operation was successful.

    Raises:
        ValueError: If the operation returns DbOperationResult with success=False.
        Exception: Any other exception that the operation itself might raise.
    """
    try:
        result = operation(*args, **kwargs)
        
        if not isinstance(result, DbOperationResult):
            raise TypeError(
                f"{operation_description} did not return a DbOperationResult object!"
            )
            
        if not result.success:
            raise ValueError(
                f"{operation_description} was not successful: {result.message}"
            )
            
        return result

    except Exception as e:
        logger.error(f"Failed during {operation_description}: {e}")
        raise e
