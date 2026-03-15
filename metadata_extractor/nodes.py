"""
Metadata extraction nodes for the LangGraph pipeline.

Architecture:
1. extraction_agent: Single LLM call to extract years and locations
2. cleaning_agent: Rule-based validation and normalization (no LLM)

This design minimizes LLM calls while ensuring high-quality output.
"""
import re
from typing import List, Set, Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models import BaseChatModel

from metadata_extractor.prompts import Prompts
from metadata_extractor.state import ExtractorAgentState
from metadata_extractor.models import ExtractionResult
from utils.logging_config import get_logger

logger = get_logger(__name__)
# Pre-compiled patterns for validation
_YEAR_PATTERN = re.compile(r'\b\d{4}\b')  # Any 4-digit number (year)
_YEAR_RANGE_PATTERN = re.compile(r'(\d{4})\s*[-–—]\s*(\d{4})')  # Year range like "2020-2023"
# Pattern for numerical data: numbers with units like °C, %, mm, km, etc.
_NUMERICAL_DATA_PATTERN = re.compile(
    r'\d+\.?\d*\s*[°%‰]|'  # Temperature, percentages
    r'\d+\.?\d*\s*(mm|cm|m|km|ft|mi)|'  # Distance/length
    r'\d+\.?\d*\s*±\s*\d+\.?\d*|'  # Uncertainty notation
    r'\d{1,3}(,\d{3})*(\.\d+)?\s*(tons?|kg|g|mt)|'  # Mass
    r'\b\d+\.\d+\s*[A-Z]\b'  # Generic number with unit (1.5 C, 3.2 K)
)


class Node:
    """
    Graph nodes for metadata extraction pipeline.
    
    Uses a single LLM call for extraction followed by rule-based
    cleaning and normalization for efficiency.
    """
    
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self._structured_llm = None
    
    @property
    def structured_llm(self):
        """Lazy initialization of structured output LLM."""
        if self._structured_llm is None:
            self._structured_llm = self.llm.with_structured_output(ExtractionResult)
        return self._structured_llm

    def extraction_agent(self, state: ExtractorAgentState) -> Dict[str, Any]:
        """
        Extract years and locations from text using LLM.
        
        This is the ONLY LLM call in the pipeline. Uses structured output
        to ensure consistent JSON formatting.
        
        Args:
            state: Contains 'text_chunk' with the text to process.
            
        Returns:
            Dict with 'raw_extraction' (ExtractionResult) or 'extraction_error'.
        """
        text_chunk = state.get('text_chunk', '')
        
        if not text_chunk or len(text_chunk.strip()) < 20:
            logger.debug("Text too short for extraction, returning empty result")
            return {
                'raw_extraction': ExtractionResult(years=[], locations=[]),
            }
        
        try:
            response = self.structured_llm.invoke([
                SystemMessage(content=Prompts.get_extractor_agent_prompt()),
                HumanMessage(content=f"Text:\n{text_chunk}")
            ])
            
            logger.debug(f"Extracted: {len(response.years)} years, {len(response.locations)} locations, {len(response.entities)} entities")
            return {'raw_extraction': response}
            
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return {
                'raw_extraction': ExtractionResult(years=[], locations=[]),
                'extraction_error': str(e)
            }

    def cleaning_agent(self, state: ExtractorAgentState) -> Dict[str, Any]:
        """
        Validate and normalize extracted metadata using rules (no LLM).
        
        Operations:
        1. Validate years against source text (prevent hallucination)
        2. Validate locations against source text
        3. Normalize location names using alias dictionary
        4. Filter invalid/generic locations
        5. Validate and normalize entities
        6. Detect numerical data (rule-based)
        7. Deduplicate and sort
        
        Args:
            state: Contains 'raw_extraction' and 'text_chunk'.
            
        Returns:
            Dict with 'clean_data' containing validated years, locations, entities, and has_numerical_data.
        """
        raw = state.get('raw_extraction')
        original_text = state.get('text_chunk', '')
        original_text_lower = original_text.lower()
        
        # Handle missing extraction
        if raw is None:
            return {'clean_data': {'years': [], 'locations': [], 'entities': [], 'has_numerical_data': False}}
        
        # === YEAR VALIDATION ===
        validated_years = self._validate_years(
            raw.years if raw.years else [],
            original_text
        )
        
        # === LOCATION VALIDATION & NORMALIZATION ===
        validated_locations = self._validate_and_normalize_locations(
            raw.locations if raw.locations else [],
            original_text_lower
        )
        
        # === ENTITY VALIDATION & NORMALIZATION ===
        validated_entities = self._validate_and_normalize_entities(
            raw.entities if raw.entities else [],
            original_text_lower
        )
        
        # === NUMERICAL DATA DETECTION (Rule-based) ===
        has_numerical_data = self._detect_numerical_data(original_text)
        
        clean_data = {
            'years': sorted(validated_years),
            'locations': sorted(validated_locations),
            'entities': sorted(validated_entities),
            'has_numerical_data': has_numerical_data
        }
        
        logger.debug(
            f"Cleaned: {len(clean_data['years'])} years, "
            f"{len(clean_data['locations'])} locations, "
            f"{len(clean_data['entities'])} entities, "
            f"numerical_data={has_numerical_data}"
        )
        return {'clean_data': clean_data}
    
    def _validate_years(self, years: List[int], source_text: str) -> Set[int]:
        """
        Validate extracted years against source text.
        
        Prevents LLM hallucination by checking if year actually appears.
        Also extracts years from ranges (e.g., "2020-2023").
        """
        validated = set()
        
        # First, find all years actually present in text
        years_in_text = set()
        for match in _YEAR_PATTERN.finditer(source_text):
            try:
                years_in_text.add(int(match.group()))
            except ValueError:
                continue
        
        # Also expand year ranges in text (e.g., "2020-2023")
        for match in _YEAR_RANGE_PATTERN.finditer(source_text):
            try:
                start_year = int(match.group(1))
                end_year = int(match.group(2))
                
                # Expand range (limit to 20 years to prevent abuse)
                if end_year >= start_year and (end_year - start_year) <= 20:
                    for y in range(start_year, end_year + 1):
                        years_in_text.add(y)
            except (ValueError, TypeError):
                continue
        
        # Validate LLM-extracted years against text
        for year in years:
            try:
                year_int = int(year)
                # Check if year appears in text OR is part of an expanded range
                if year_int in years_in_text:
                    validated.add(year_int)
            except (ValueError, TypeError):
                continue
        
        return validated
    
    def _validate_and_normalize_locations(
        self, 
        locations: List[str], 
        source_text_lower: str
    ) -> Set[str]:
        """
        Validate locations against source text and normalize names.
        
        Steps:
        1. Check if location (or variant) appears in source text
        2. Filter out invalid entries (organizations, phenomena, etc.)
        3. Normalize using alias dictionary
        4. Apply consistent formatting
        """
        validated = set()
        
        for loc in locations:
            if not loc:
                continue
                
            loc_clean = str(loc).strip()
            loc_lower = loc_clean.lower()
            
            # Skip if too short or too long
            if len(loc_clean) < 2 or len(loc_clean) > 60:
                continue
            
            # Skip invalid locations (organizations, phenomena, etc.)
            if loc_lower in Prompts.INVALID_LOCATIONS:
                continue
            
            # Skip if contains obvious junk
            if any(junk in loc_lower for junk in ['note:', 'figure', 'table', 'source:', 'http']):
                continue
            
            # Validate: location must appear in source text
            # Check both the raw name and common variations
            appears_in_text = (
                loc_lower in source_text_lower or
                loc_clean in source_text_lower or
                # Also check normalized form
                Prompts.LOCATION_ALIASES.get(loc_lower, '').lower() in source_text_lower
            )
            
            if not appears_in_text:
                # Try partial match for multi-word locations
                words = loc_lower.split()
                if len(words) > 1:
                    appears_in_text = any(
                        word in source_text_lower 
                        for word in words 
                        if len(word) > 3
                    )
            
            if not appears_in_text:
                logger.debug(f"Location '{loc_clean}' not found in source text, skipping")
                continue
            
            # Normalize using alias dictionary
            normalized = Prompts.LOCATION_ALIASES.get(loc_lower, loc_clean)
            
            # Apply title case if not already properly formatted
            if normalized == loc_clean and not any(c.isupper() for c in normalized):
                normalized = normalized.title()
            
            # Remove common suffixes that add noise
            for suffix in [' Province', ' Region', ' District', ' State']:
                if normalized.endswith(suffix) and len(normalized) > len(suffix) + 3:
                    # Keep if it's a well-known name with the suffix
                    base = normalized[:-len(suffix)]
                    if base.lower() not in Prompts.LOCATION_ALIASES:
                        normalized = base
            
            validated.add(normalized)
        
        return validated
    
    def _validate_and_normalize_entities(
        self,
        entities: List[str],
        source_text_lower: str
    ) -> Set[str]:
        """
        Validate entities against source text and normalize names.
        
        Entities are organizations, institutions, agencies that are
        mentioned in the text. Unlike locations, we want to keep these
        even if they were filtered from locations.
        """
        validated = set()
        
        for entity in entities:
            if not entity:
                continue
            
            entity_clean = str(entity).strip()
            entity_lower = entity_clean.lower()
            
            # Skip if too short or too long
            if len(entity_clean) < 2 or len(entity_clean) > 60:
                continue
            
            # Skip obvious junk
            if any(junk in entity_lower for junk in ['note:', 'figure', 'table', 'http']):
                continue
            
            # Validate: entity must appear in source text
            appears_in_text = (
                entity_lower in source_text_lower or
                entity_clean in source_text_lower or
                # Check normalized form
                Prompts.ENTITY_ALIASES.get(entity_lower, '').lower() in source_text_lower
            )
            
            if not appears_in_text:
                # Try checking if acronym matches
                if entity_clean.isupper() and len(entity_clean) <= 6:
                    # For acronyms, check if it appears as a word boundary
                    if re.search(rf'\b{re.escape(entity_clean)}\b', source_text_lower, re.IGNORECASE):
                        appears_in_text = True
            
            if not appears_in_text:
                logger.debug(f"Entity '{entity_clean}' not found in source text, skipping")
                continue
            
            # Normalize using alias dictionary
            normalized = Prompts.ENTITY_ALIASES.get(entity_lower, entity_clean)
            
            # Keep uppercase for known acronyms
            if len(normalized) <= 6 and normalized.isupper():
                validated.add(normalized)
            else:
                # Apply title case for full names
                if normalized == entity_clean and not any(c.isupper() for c in normalized):
                    normalized = normalized.title()
                validated.add(normalized)
        
        return validated
    
    def _detect_numerical_data(self, text: str) -> bool:
        """
        Detect if text contains numerical data with units.
        
        Rule-based detection for measurements, statistics, etc.
        Useful for boosting chunks that contain actual data vs. narrative.
        """
        return bool(_NUMERICAL_DATA_PATTERN.search(text))