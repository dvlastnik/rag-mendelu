from pathlib import Path
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain_community.llms.gpt4all import GPT4All

from database.base.BaseDbRepository import BaseDbRepository
from text_embedding_api.TextEmbeddingService import TextEmbeddingService
from rag.Retriever import Retriever

class RAG:
    def __init__(self, database_service: BaseDbRepository, embedding_service: TextEmbeddingService, model_path: str):
        self._check_llm_path(llm_path=model_path)

        self.llm = GPT4All(
            model=model_path,
            allow_download=False,
            n_threads=8,
            device="cpu",   # "cuda" for NVIDIA GPU or "cpu" to run only on cpu
            n_batch=64
        )
        self.database_service = database_service
        self.embedding_service = embedding_service

        self.qa = RetrievalQAWithSourcesChain.from_chain_type(llm=self.llm, retriever=Retriever(database_service=self.database_service, embedding_service=self.embedding_service), chain_type="stuff")

    def _check_llm_path(self, llm_path: str):
        p = Path(llm_path)

        if not p.exists():
            raise FileNotFoundError(f"Path for llm model is invalid! {llm_path}")
        if not p.is_file():
            raise IsADirectoryError(f"File is excepted but got path: {llm_path}")
        if p.suffix.lower() != ".gguf":
            raise ValueError(f"Model file must have .gguf extension: {llm_path}")
        
    def chat(self, question: str):
        result = self.qa.invoke({"question": question})
        print(f"Answer: {result['answer']}")
        print(f"Sources: {result['sources']}")
        

