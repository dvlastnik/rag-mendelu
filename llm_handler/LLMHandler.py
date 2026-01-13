from typing import Dict, List, Type, Any, Union
import ollama
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel

from utils.logging_config import get_logger
from utils.singleton_wrapper import singleton

logger = get_logger(__name__)

@singleton
class LLMHandler:
    def __init__(self, ip: str = "127.0.0.1", port: int = 11434):
        self.base_url = f"http://{ip}:{port}"
        
        try:
            self._client = ollama.Client(host=self.base_url)
            self._client.list()
            logger.info(f"Ollama connected at {self.base_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Ollama at {self.base_url}: {e}")
            raise ValueError(f"Ollama Connection Error: {e}")

        self.active_model_name: str | None = None
        self._langchain_llm: ChatOllama | None = None
        self.history: List[BaseMessage] = []

    def load_model(self, model_name: str, temperature: float = 0.0):
        """
        Prepares the model. Checks if it exists locally; pulls if not.
        Initializes the LangChain ChatOllama instance.
        """
        logger.debug(f"Selecting model: {model_name}")

        self.active_model_name = model_name
        self._langchain_llm = ChatOllama(
            model=model_name,
            base_url=self.base_url,
            temperature=temperature,
            keep_alive="5m"
        )
        logger.info(f"Model {model_name} is ready for inference.")

    def unload_model(self):
        """
        Forces the model out of VRAM immediately using the Ops client.
        """
        if self.active_model_name:
            logger.debug(f"Unloading {self.active_model_name}...")
            try:
                self._client.chat(model=self.active_model_name, messages=[], keep_alive=0)
                self.active_model_name = None
                self._langchain_llm = None
                logger.info("Model unloaded from VRAM.")
            except Exception as e:
                logger.warning(f"Unload failed (server might be down): {e}")

    def get_llm(self) -> ChatOllama:
        if not self._langchain_llm:
            raise RuntimeError("No model loaded. Call load_model() first.")
        return self._langchain_llm

    def chat_and_get_response(
        self, 
        user_prompt: str, 
        system_prompt: str | None = None, 
        response_scheme: Type[BaseModel] | None = None
    ) -> Union[str, BaseModel]:
        """
        Stateless (One-off) generation.
        Supports Pydantic Structured Output.
        """
        llm = self.get_llm()
        messages = []

        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=user_prompt))

        try:
            if response_scheme:
                logger.debug(f"Invoking with Structured Output: {response_scheme.__name__}")
                structured_llm = llm.with_structured_output(response_scheme)
                return structured_llm.invoke(messages)

            logger.debug("Invoking standard chat...")
            response = llm.invoke(messages)
            return response.content

        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise e

    def set_chat(self, system_prompt: str):
        """
        Resets the internal history for stateful chat.
        """
        self.history = [SystemMessage(content=system_prompt)]
        logger.debug("Chat history reset.")

    def ask_and_get_response(
        self, 
        user_prompt: str, 
        response_scheme: Type[BaseModel] | None = None
    ) -> Union[str, BaseModel]:
        """
        Stateful chat. Appends to history.
        NOTE: Structured output is rarely added to history as 'text', 
        so we handle it carefully.
        """
        if not self.history:
            logger.warning("History empty. Starting fresh.")
            self.history = []

        self.history.append(HumanMessage(content=user_prompt))
        llm = self.get_llm()

        try:
            if response_scheme:
                structured_llm = llm.with_structured_output(response_scheme)
                result_obj = structured_llm.invoke(self.history)
                
                json_repr = result_obj.model_dump_json()
                self.history.append(AIMessage(content=json_repr))
                
                return result_obj
            else:
                response_msg = llm.invoke(self.history)
                self.history.append(response_msg)
                return response_msg.content

        except Exception as e:
            logger.error(f"Stateful chat failed: {e}")
            raise e