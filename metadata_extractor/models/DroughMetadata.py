import lmstudio as lms
from typing import List

DROUGH_METADATA_PROMPT = """
    Extract the required metadata from the following text based on the schema's rules.

    RULES:
    - **topics**: Identify 1-3 main topics. Topics must be 1-3 words, lowercase.
    - **years**: Extract *all* 4-digit years.
    - **countries**: Extract *only* country names.
    - If no items are found for a category, return an empty list [].
    - Do not make up information.

    Text to Analyze:
    "{input_text}"
"""

class DroughMetadata(lms.BaseModel):
    countries: List[str]
    years: List[str]
    topics: List[str]