class Prompts:
   # Location normalization mappings (rule-based, no LLM needed)
   LOCATION_ALIASES = {
      # Country aliases
      'uk': 'United Kingdom',
      'u.k.': 'United Kingdom',
      'great britain': 'United Kingdom',
      'britain': 'United Kingdom',
      'england': 'United Kingdom',  # Debatable, but common in climate docs
      'usa': 'United States',
      'u.s.a.': 'United States',
      'u.s.': 'United States',
      'us': 'United States',
      'america': 'United States',
      'united states of america': 'United States',
      'prc': 'China',
      "people's republic of china": 'China',
      'rok': 'South Korea',
      'dprk': 'North Korea',
      'uae': 'United Arab Emirates',
      'swiss': 'Switzerland',
      'holland': 'Netherlands',
      'burma': 'Myanmar',
      'persia': 'Iran',
      'viet nam': 'Vietnam',
      
      # Region standardization
      'arctic ocean': 'Arctic',
      'the arctic': 'Arctic',
      'antarctic': 'Antarctica',
      'the antarctic': 'Antarctica',
      'sub-saharan africa': 'Sub-Saharan Africa',
      'subsaharan africa': 'Sub-Saharan Africa',
      'middle east': 'Middle East',
      'mideast': 'Middle East',
      'southeast asia': 'Southeast Asia',
      'south-east asia': 'Southeast Asia',
      'se asia': 'Southeast Asia',
      'east asia': 'East Asia',
      'western europe': 'Western Europe',
      'eastern europe': 'Eastern Europe',
      'central europe': 'Central Europe',
      'north america': 'North America',
      'south america': 'South America',
      'latin america': 'Latin America',
      'central america': 'Central America',
      
      # Ocean/Sea standardization
      'med': 'Mediterranean Sea',
      'mediterranean': 'Mediterranean Sea',
      'atlantic': 'Atlantic Ocean',
      'pacific': 'Pacific Ocean',
      'indian ocean': 'Indian Ocean',
      'southern ocean': 'Southern Ocean',
   }

   # Locations to filter out (not actual geographic locations)
   INVALID_LOCATIONS = {
      # Organizations mistakenly extracted
      'wmo', 'ipcc', 'fao', 'un', 'united nations', 'noaa', 'nasa',
      'ecmwf', 'copernicus', 'esa', 'eumetsat', 'jma', 'cma',
      
      # Climate phenomena
      'enso', 'el nino', 'el niño', 'la nina', 'la niña',
      'nao', 'pdo', 'amo', 'iod', 'mjo',
      
      # Generic terms
      'global', 'international', 'worldwide', 'world', 'earth',
      'hemisphere', 'northern hemisphere', 'southern hemisphere',
      'tropics', 'tropical', 'equator', 'equatorial',
      
      # Document artifacts
      'figure', 'table', 'source', 'note', 'reference',
   }

   # Known entity abbreviations for normalization
   ENTITY_ALIASES = {
      # Climate/Weather organizations
      'world meteorological organization': 'WMO',
      'intergovernmental panel on climate change': 'IPCC',
      'food and agriculture organization': 'FAO',
      'united nations': 'UN',
      'world health organization': 'WHO',
      
      # Space agencies
      'national aeronautics and space administration': 'NASA',
      'european space agency': 'ESA',
      'national oceanic and atmospheric administration': 'NOAA',
      'japan aerospace exploration agency': 'JAXA',
      
      # Research/Forecast centers
      'european centre for medium-range weather forecasts': 'ECMWF',
      'national center for atmospheric research': 'NCAR',
      
      # Multi-word standardization
      'world bank': 'World Bank',
      'european union': 'EU',
   }

   @staticmethod
   def get_extractor_agent_prompt() -> str:
        return """Extract years, locations, and entities from climate text.

YEARS: Extract any 4-digit number as a year. Expand ranges: "2020-2023" → [2020,2021,2022,2023], "2010-12" → [2010,2011,2012].

LOCATIONS: Countries, regions, cities, water bodies (e.g., Pakistan, East Africa, Mediterranean).
- Skip: organizations (WMO, NASA), people names, climate terms (ENSO, La Niña), generic words (Global).

ENTITIES: Organizations mentioned (WMO, IPCC, NASA, UN, FAO, NOAA, ESA).

Examples:
"2022 floods in Pakistan by WMO" → {"years":[2022], "locations":["Pakistan"], "entities":["WMO"]}
"Drought since 2020, like 2010-12" → {"years":[2020,2010,2011,2012], "locations":[], "entities":[]}
"Antarctic ice loss per NASA, ESA" → {"years":[], "locations":["Antarctic"], "entities":["NASA","ESA"]}
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