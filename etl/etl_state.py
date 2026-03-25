from enum import Enum

class ETLState(Enum):
    NOT_STARTED = 1
    EXTRACTED = 2
    TRANSFORMED = 3
    LOADED = 4
    FAILED = 5
    FILE_NOT_FOUND = 6