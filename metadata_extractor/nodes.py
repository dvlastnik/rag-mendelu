import re
import json
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.chat_models import BaseChatModel

from metadata_extractor.prompts import Prompts
from metadata_extractor.state import ExtractorAgentState
from metadata_extractor.models import ExtractionResult

class Node:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def extraction_agent(self, state: ExtractorAgentState):
        structured_llm = self.llm.with_structured_output(ExtractionResult)
        
        response = structured_llm.invoke([
            SystemMessage(content=Prompts.get_extractor_agent_prompt()),
            HumanMessage(content=f"Text: {state['text_chunk']}")
        ])

        return {'raw_extraction': response}
    
    def normalization_agent(self, state: ExtractorAgentState):
        raw_data = state['raw_extraction']
        
        structured_llm = self.llm.with_structured_output(ExtractionResult)
        raw_json_str = json.dumps(raw_data.model_dump())
        
        response = structured_llm.invoke([
            SystemMessage(content=Prompts.get_normalization_agent_prompt()),
            HumanMessage(content=raw_json_str)
        ])
        return {'final_extraction': response}

    def cleaning_agent(self, state: ExtractorAgentState):
        raw = state['final_extraction']
        original_text = state['text_chunk']
        original_text_lower = original_text.lower()
        
        clean_obj = {
            'years': [],
            'locations': []
        }
        
        year_pattern = re.compile(r'^(19|20)\d{2}$')
        if raw.years:
            unique_years = set()
            for y in raw.years:
                try:
                    y_int = int(y)
                    y_str = str(y_int)
                    if not year_pattern.match(y_str):
                        continue
                    
                    if y_str not in original_text:
                        continue
                    
                    unique_years.add(y_int)
                except ValueError:
                    continue
                    
            clean_obj['years'] = sorted(list(unique_years))

        if raw.locations:
            unique_countries = set()
            for c in raw.locations:
                c_str = str(c).strip()
                if len(c_str) < 2 or len(c_str) > 50 or 'note' in c_str.lower():
                    continue
                
                unique_countries.add(c_str.title())

            clean_obj['locations'] = sorted(list(unique_countries))
        
        return {'clean_data': clean_obj}