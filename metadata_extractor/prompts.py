class Prompts:
    @staticmethod
    def get_extractor_agent_prompt() -> str:
        return """### TASK
You are metadata extractor from the text below. Extract all physical location and also any mentioned years.

### TARGETS
1. **Countries** (e.g., France, China, USA)
2. **Regions/Continents** (e.g., East Africa, Europe, Antarctica)
3. **Cities/States** (e.g., Paris, Texas, Balochistan)
4. **Oceans/Seas** (e.g., Mediterranean Sea)

### RULES
- **Extract** full names (e.g., "United Kingdom", not just "Kingdom").
- **IGNORE** Organizations (UN, FAO, IPCC, WMO).
- **IGNORE** People/Authors (Smith, J., Doe).
- **IGNORE** Citations and References.

### EXAMPLES
Input: "2020 Floods in Pakistan and the UK caused by WMO reports."
Output: {"years": [2020], "locations": ["Pakistan", "UK"]}

Input: "Study by Smith et al. (2020) in the Amazon Basin. 2020-2023"
Output: {"years": [2020, 2023], "locations": ["Amazon Basin"]}
"""
    
    @staticmethod
    def get_normalization_agent_prompt() -> str:
        return """### TASK
Clean and Standardize this list of locations.

### CHECKLIST
1. **Remove Non-Locations:**
   - DELETE Agencies (FAO, WMO, IPCC, NOAA, NASA).
   - DELETE Generic words (Global, International, World).
   - DELETE Climate terms (ENSO, La Nina, Southern Oscillation).

2. **Standardize Names:**
   - "UK" / "Great Britain" -> "United Kingdom"
   - "USA" / "America" -> "United States"
   - "Swiss" -> "Switzerland"
   - "Viet Nam" -> "Vietnam"

3. **Fix Formatting:**
   - Fix Case: "east africa" -> "East Africa"
   - Remove "Province" (e.g. "Sindh Province" -> "Sindh")

### INPUT DATA
"""