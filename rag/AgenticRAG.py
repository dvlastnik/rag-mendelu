from typing import Dict, Any
import traceback

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from database.base.BaseDbRepository import BaseDbRepository
from text_embedding import TextEmbeddingService
from rag.agents.graph import build_graph
from utils.logging_config import get_logger

logger = get_logger(__name__)

class AgenticRAG:
    def __init__(
        self,
        database_service: BaseDbRepository,
        embedding_service: TextEmbeddingService,
        model_name: str = "llama3.1:8b",
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
            'rewritten_queries': [],
            'search_results': [],
            'extracted_data': [],
            'filtered_results': [],
            'hallucination_status': None
        }
        
        try:
            final_state = self.agents.invoke(initial_state, config={"recursion_limit": 50})
            last_message = final_state['messages'][-1]

            rewritten_queries = final_state.get('rewritten_queries', [])
            return {
                'agent_state': final_state,
                'original_query': question,
                'rewritten_queries': rewritten_queries,
                'extracted_data': final_state['extracted_data'],
                'response': last_message.content,
                'sources': final_state['filtered_results'],
                'compressor_results': final_state['context_compressor_results']
            }
            
        except Exception as e:
            traceback.print_exc()
            logger.error(f"Agent Workflow Failed: {e}")
            return {
                'agent_state': initial_state,
                'response': f"I encountered an error while processing your request: {str(e)}",
                'sources': []
            }