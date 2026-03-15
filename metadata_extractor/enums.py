from enum import Enum


class NodeName(str, Enum):
    """Node names for the metadata extraction graph."""
    EXTRACTOR = 'extractor_agent'
    CLEAN = 'clean_agent'  # Combined validation + normalization