from operator import itemgetter
from typing import List, Dict, Any, Optional

from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document

from database.base.BaseDbRepository import BaseDbRepository
from text_embedding_api.TextEmbeddingService import TextEmbeddingService
from rag.HybridRetriever import HybridRetriever
from llm_handler.LLMHandler import LLMHandler
from utils.logging_config import get_logger

logger = get_logger(__name__)

def _format_docs(docs: List[Document]) -> str:
    """
    Combines the page content of retrieved documents into a single string.
    """
    return "\n\n".join(doc.page_content for doc in docs)

class RAG:
    def __init__(
        self, 
        database_service: BaseDbRepository, 
        embedding_service: TextEmbeddingService,
        llm_handler: LLMHandler,
        model_name: str = "llama3.2:3b"
    ):  
        logger.info("Initializing Hybrid Retriever...")
        self.retriever = HybridRetriever(
            database_service=database_service, 
            embedding_service=embedding_service,
            k=3
        )

        logger.info("Resolving LLM...")
        
        logger.info("Using provided LLMHandler instance.")
        try:
            self.llm = llm_handler.get_llm()
        except RuntimeError:
            logger.warning(f"LLMHandler had no active model. Loading '{model_name}'...")
            llm_handler.load_model(model_name)
            self.llm = llm_handler.get_llm()

        logger.info("Building RAG chain...")
        
        rag_prompt = ChatPromptTemplate.from_template(
            """You are an assistant for question-answering tasks.
            Use the following pieces of retrieved context to answer the question.
            If you don't know the answer from the context, just say "I don't know."
            Keep your answer concise.

            Context:
            {context}

            Question:
            {question}

            Answer:
            """
        )

        setup_and_retrieval = RunnableParallel(
            {
                "context": self.retriever, 
                "question": RunnablePassthrough()
            }
        )

        answer_generation = (
            {
                "context": itemgetter("context") | RunnableLambda(_format_docs),
                "question": itemgetter("question")
            }
            | rag_prompt
            | self.llm 
            | StrOutputParser()
        )

        self.qa_chain = (
            setup_and_retrieval 
            | RunnableParallel({
                "answer": answer_generation,
                "sources": itemgetter("context")
            })
        )
        
        logger.info("RAG chain built successfully.")

    def chat(self, question: str) -> Dict[str, Any]:
        """
        Runs the RAG chain on a given question.
        
        Args:
            question: The user's query string.
            
        Returns:
            Dict containing:
            - 'answer': The generated string response.
            - 'sources': List[Document] objects used for context.
        """
        logger.info(f"Invoking chain with question: '{question}'")
        result = self.qa_chain.invoke(question)
        return result