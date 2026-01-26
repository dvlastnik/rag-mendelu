from typing import List, Dict, Any

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from database.base.BaseDbRepository import BaseDbRepository
from text_embedding_api.TextEmbeddingService import TextEmbeddingService
from rag.agents.graph import build_graph
from utils.logging_config import get_logger

logger = get_logger(__name__)

class AgenticRAG:
    def __init__(
        self, 
        database_service: BaseDbRepository, 
        embedding_service: TextEmbeddingService,
        model_name: str = "llama3.1:8b"
    ):  
        self.agents = build_graph(database_service, embedding_service, model_name)
        

    def chat(self, question: str) -> Dict[str, Any]:
        """
        Main entry point. 
        Converts string input -> AgentState -> Invokes Graph -> Returns final string.
        """
        logger.info(f"--- STARTING AGENT WORKFLOW FOR: '{question}' ---")
        
        initial_state = {
            'messages': [HumanMessage(content=question)],
            'rewritten_query': '',
            'search_results': [],
            'extracted_data': [],
            'filtered_results': [],
            'hallucination_status': None
        }
        
        try:
            final_state = self.agents.invoke(initial_state, config={"recursion_limit": 50})
            last_message = final_state['messages'][-1]

            return {
                'response': last_message.content,
                'sources': final_state['filtered_results']
            }
            
        except Exception as e:
            logger.error(f"Agent Workflow Failed: {e}")
            return f"I encountered an error while processing your request: {str(e)}"