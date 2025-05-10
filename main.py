import chromadb
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from gpt4all import GPT4All
from datetime import datetime
import logging
import argparse
import os

from ChromaDbModel import ChromaDbModel
from ChromaService import ChromaDbService
from TextEmbeddingService import TextEmbeddingService



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('chromadb')

TEXT_EMBEDDING_URL='http://127.0.0.1:8000'
CHROMADB_URL='127.0.0.1:8001'
COLLECTION_NAME = 'witcher_books'

def load_documents(documents_path: str):
    if not os.path.exists(documents_path):
        raise FileNotFoundError(f"Directory not found: {documents_path}")
    if not os.path.isdir(documents_path):
        raise NotADirectoryError(f"Entered path is not directory {documents_path}")
    
    loader = DirectoryLoader(documents_path, glob="*.md", loader_cls=UnstructuredMarkdownLoader)
    documents = loader.load()
    return documents

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True
    )

    chunks = text_splitter.split_documents(documents=documents)
    print(f"Splitted {len(documents)} into {len(chunks)} chunks.")

    return chunks

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process arguments.")
    
    parser.add_argument('--init-db', action='store_true', help='Initialize chromadb')
    parser.add_argument('--prompt', type=str, help='Prompt for llm', default=None)
    args = parser.parse_args()

    return args

def initialize_chromadb(embedding_service: TextEmbeddingService, chroma_service: ChromaDbService):
    chroma_service.create_collection(
        collection_name=COLLECTION_NAME, 
        metadata={
            'description': 'Collection containing information about witcher books',
            'created': str(datetime.now())
        }
    )

    docs = load_documents('data/witcher_md')
    chunks = split_documents(docs)

    chroma_db_models: list[ChromaDbModel] = []
    for chunk in chunks:
        chroma_db_models.append(
            ChromaDbModel.from_document(chunk, embedding_service)
        )

    chroma_service.add_data(data=chroma_db_models)

    print(chroma_service.collection.peek(limit=3))

def ask_local_llm(query: str, embedding_service: TextEmbeddingService, chroma_service: ChromaDbService):
    query_embedding = embedding_service.get_embedding_with_uuid([query])

    chroma_service.get_collection(collection_name=COLLECTION_NAME)
    result = chroma_service.query(query=query_embedding[0].embedding)
    context = chroma_service.from_query_result_to_context_str(result=result)

    prompt = f"""You are an expert on The Witcher books.
    Answer the question based on the context below.

    Context:
    {context}

    Question: {query}
    Answer:"""
    print(prompt)

    model = GPT4All(
        model_name="Meta-Llama-3-8B-Instruct.Q4_0.gguf",
        model_path=r"C:\Users\david\AppData\Local\nomic.ai\GPT4All",
        allow_download=False
    )
    with model.chat_session():
        response = model.generate(prompt, max_tokens=512, temp=0.7)
        print(response)

def main():
    embedding_service = TextEmbeddingService(TEXT_EMBEDDING_URL)
    chroma_service = ChromaDbService(CHROMADB_URL)

    args = parse_arguments()
    if args.init_db:
        initialize_chromadb(embedding_service, chroma_service)

    if args.prompt:
        # When and how did Geralt meet Regis and in which book?
        ask_local_llm(
            query=args.prompt,
            embedding_service=embedding_service,
            chroma_service=chroma_service
        )

if __name__ == '__main__':
    main()
    

    

