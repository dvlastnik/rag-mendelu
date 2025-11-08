from pathlib import Path
from operator import itemgetter
from langchain_community.llms.gpt4all import GPT4All
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document

from database.base.BaseDbRepository import BaseDbRepository
from text_embedding_api.TextEmbeddingService import TextEmbeddingService
from rag.Retriever import Retriever
from utils.logging_config import get_logger

logger = get_logger(__name__)

def _format_docs(docs: list[Document]) -> str:
    """Combines all document page_content into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

def _get_sources(docs: list[Document]) -> list[dict]:
    """Extracts the metadata from the retrieved documents."""
    # Returns the full metadata. You could also just return a list
    # of doc.metadata.get("source", "Unknown") if you only want the file path.
    return [doc.metadata for doc in docs]

class RAG:
    def __init__(self, database_service: BaseDbRepository, embedding_service: TextEmbeddingService, model_path: str):
        self._check_llm_path(llm_path=model_path)

        logger.info("Initializing LLM...")
        self.llm = GPT4All(
            model=model_path,
            allow_download=False,
            n_threads=8,
            device="cpu",
            n_batch=64
        )
        
        logger.info("Initializing Retriever...")
        self.retriever = Retriever(
            database_service=database_service, 
            embedding_service=embedding_service,
            k=3
        )

        logger.info("Building RAG chain...")
        # Define the prompt template for the LLM
        RAG_PROMPT_TEMPLATE = """
        You are an assistant for question-answering tasks.
        Use the following pieces of retrieved context to answer the question.
        If you don't know the answer from the context, just say "I don't know."
        Keep your answer concise.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
        prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

        # Create the LCEL RAG chain
        # This chain is designed to return both the answer and the source documents
        
        # 1. This runnable retrieves context and passes the question through
        setup_and_retrieval = RunnableParallel(
            {"context": self.retriever, "question": RunnablePassthrough()}
        )

        # 2. This runnable formats the prompt and generates the answer
        answer_generation = {
            "answer": (
                {
                    "context": itemgetter("context") | RunnableLambda(_format_docs),
                    "question": itemgetter("question")
                }
                | prompt
                | self.llm
                | StrOutputParser()
            ),
            "sources": itemgetter("context") # Pass the original Document objects through
        }

        # 3. Combine them into the final chain
        self.qa_chain = setup_and_retrieval | answer_generation
        
        logger.info("RAG chain built successfully.")


    def _check_llm_path(self, llm_path: str):
        p = Path(llm_path)
        if not p.exists():
            raise FileNotFoundError(f"Path for llm model is invalid! {llm_path}")
        if not p.is_file():
            raise IsADirectoryError(f"File is excepted but got path: {llm_path}")
        if p.suffix.lower() != ".gguf":
            raise ValueError(f"Model file must have .gguf extension: {llm_path}")
        
    def chat(self, question: str) -> dict:
        """
        Runs the RAG chain on a given question.
        
        Args:
            question: The user's question.
            
        Returns:
            A dictionary containing:
            - "answer": The LLM's generated answer.
            - "sources": A list of Document objects used as context.
        """
        logger.info(f"Invoking chain with question: '{question}'")
        result = self.qa_chain.invoke(question)
        return result
        

