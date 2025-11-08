import lmstudio as lms
from typing import Dict, List

from utils.logging_config import get_logger
from utils.singleton_wrapper import singleton

logger = get_logger(__name__)

@singleton
class LLMHandler:
    def __init__(self, ip: str, port: int):
        api_host = ip + ":" + str(port)
        lms.configure_default_client(api_host)
        if lms.Client.is_valid_api_host(api_host):
            logger.debug(f"LM Studio API server instance is available at {api_host}")
        else:
            logger.error(f"No LM Studio API server instance found at {api_host}")
            raise ValueError(f"No LM Studio API server instance found at {api_host}")
        
        self.model: lms.LLM = None
        self.chat: lms.Chat = None

    def _respond(self, chat: lms.Chat, response_scheme: lms.ResponseSchema | None = None) -> str:
        chat_response = ''
        if response_scheme is not None:
            chat_response = self.model.respond(chat, response_format=response_scheme).parsed
        else:
            chat_response = self.model.respond(chat).content

        logger.debug(f"Chat response: {chat_response}")

        return chat_response

    def load_model(self, model_name: str, config: Dict | None = None):
        if self.model is None:
            logger.debug(f"Loading model: {model_name}")

            if config is not None:
                self.model = lms.llm(model_name, **config)
            else:
                self.model = lms.llm(model_name)

            logger.debug(f"Model {model_name} loaded.")
        else:
            logger.debug(f"Some model is already loaded {self.model.get_info()}")

    def unload_model(self):
        if self.model:
            logger.debug(f"Unloading model: {self.model.identifier}")
            self.model.unload()
            self.model = None
            logger.debug(f"Model {self.model.identifier} unloaded")

    def get_model(self) -> lms.LLM:
        if self.model is None:
            logger.error("Attempted to get model, but it is not loaded.")
            raise RuntimeError(
                "Model is not loaded. "
                "Call .load() first or use the handler as a context manager."
            )
        return self.model
    
    def chat_and_get_response(self, user_prompt: str | List[str], system_prompt: str | None = None, response_scheme: lms.ResponseSchema | None = None) -> str:
        if system_prompt is None or system_prompt == '':
            logger.error(f"Intitial prompt is empty")
            raise RuntimeError(f"Initial prompt is empty")
        
        chat = lms.Chat(system_prompt)
        chat.add_user_message(user_prompt)
        
        return self._respond(chat=chat, response_scheme=response_scheme)
    
    def set_chat(self, system_prompt: str):
        self.chat = lms.Chat(initial_prompt=system_prompt)

    def ask_and_get_response(self, user_prompt: str | List[str], response_scheme: lms.ResponseSchema | None = None) -> str:
        self.chat.add_user_message(user_prompt)

        return self._respond(chat=self.chat, response_scheme=response_scheme)