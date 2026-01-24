from typing import List
import pydantic

DROUGH_METADATA_PROMPT = """
You are an advanced entity extraction system. Analyze the input text and extract specific metadata according to the RULES below.

### RULES:
1. **Countries**: 
   - Extract all mentioned countries.
   - **CRITICAL:** If a city is mentioned but its country is NOT, you MUST infer the country (e.g., "Tokyo" -> add "Japan").
   - **NORMALIZE:** Always convert acronyms to full official names (e.g., "USA" -> "United States", "UK" -> "United Kingdom").

2. **Cities**:
   - Extract all mentioned cities.
   - **NORMALIZE:** Expand common abbreviations (e.g., "NYC" -> "New York City", "LA" -> "Los Angeles").

3. **Years**:
   - Extract all mentioned years.

3. **Exclusions**:
   - Ignore names of people even if they match location names (e.g., ignore "Jordan" or "Chelsea" if they refer to people).

4. **Output Format**:
   - Return valid JSON matching the schema.
   - If nothing is found, return empty lists [].

### Input Text:
"{input_text}"
"""

class DroughMetadata(pydantic.BaseModel):
   countries: List[str] = pydantic.Field(
        default_factory=list, 
        description="List of countries mentioned in the text."
    )
   cities: List[str] = pydantic.Field(
        default_factory=list, 
        description="List of cities mentioned in the text."
    )
   years: List[str] = pydantic.Field(
        default_factory=list, 
        description="List of specific years mentioned (e.g., '2022', '1990-2000')."
    )