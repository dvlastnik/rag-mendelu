import lmstudio as lms
from typing import List

from utils.logging_config import get_logger
from llm_handler.LLMHandler import LLMHandler

logger = get_logger(__name__)

class LLMMetadataExtractor:
    SYSTEM_PROMPT = "You are an expert metadata extraction service. You will be given text and a JSON schema. Your sole purpose is to analyze the text, extract the required entities, and populate the JSON schema. Respond *only* with the JSON object."

    def __init__(self, llm_handler: LLMHandler):
        self.llm_handler = llm_handler

        # Set model for extracting
        self.llm_handler.unload_model()
        self.llm_handler.load_model("llama-3.2-3b-instruct")

    def extract_metadata(self, prompt: str | List[str], response_scheme: lms.ResponseSchema) -> str:
        return self.llm_handler.chat_and_get_response(user_prompt=prompt, system_prompt=self.SYSTEM_PROMPT, response_scheme=response_scheme)