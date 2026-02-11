from enum import Enum

class NodeName(str, Enum):
    EXTRACTOR = 'extractor_agent'
    NORMALIZATION = 'normalization_agent'
    CLEAN = 'clean_agent'