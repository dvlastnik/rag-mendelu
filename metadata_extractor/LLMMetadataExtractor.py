from typing import List, Type, TypeVar, Union
from pydantic import BaseModel

from utils.logging_config import get_logger
from llm_handler.LLMHandler import LLMHandler

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

class LLMMetadataExtractor:
    SYSTEM_PROMPT = (
        "You are an expert metadata extraction service. "
        "Analyze the input text and extract the entities defined in the schema."
    )

    def __init__(self, llm_handler: LLMHandler):
        self.llm_handler = llm_handler
        self.model_name = "llama3.2:3b" 
        self.llm_handler.load_model(self.model_name)

    def extract_metadata(self, prompt: str, response_scheme: Type[T]) -> T:
        """
        Sends text to the LLM and returns a validated Pydantic object.
        """
        result = self.llm_handler.chat_and_get_response(
            user_prompt=prompt, 
            system_prompt=self.SYSTEM_PROMPT, 
            response_scheme=response_scheme
        )
        
        return result