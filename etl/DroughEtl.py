import json
import traceback
from typing import List, Tuple, Optional, Dict
import pandas as pd
import pathlib
import re
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
import lmstudio as lms
import sys

from database.ChromaDbRepository import ChromaDbRepository
from etl.BaseEtl import BaseEtl, ETLState
from database.base.MyDocument import MyDocument
from text_embedding_api.TextEmbeddingService import ChunkAndEmbedResponse
from metadata_extractor.LLMMetadataExtractor import LLMMetadataExtractor
from metadata_extractor.models import DroughMetadata
from utils.logging_config import get_logger

logger = get_logger(__name__)

class DroughEtl(BaseEtl):
    def __init__(self, filepath, db_repository, embedding_service, metadata_extractor: LLMMetadataExtractor):
        super().__init__(filepath, db_repository, embedding_service)
        self.metadata_extractor = metadata_extractor

    def _row_to_document(self, row):
        return super()._row_to_document(row)

    def get_file_path(self, only_folder: bool = False, chunk_index: int | None = None) -> pathlib.Path:
        folder_prefix = "data/ignore_"

        if only_folder:
            return pathlib.Path(f"{folder_prefix}{self.file.stem}")
        if chunk_index is not None:
            return pathlib.Path(f"{folder_prefix}{self.file.stem}/{self.file.stem}_{chunk_index}.md")
        return pathlib.Path(f"{folder_prefix}{self.file.stem}/{self.file.stem}.md")

    def _merge_body(self, body: str) -> str:
        if "table" in body.lower() or "table of contents" not in body.lower():
            return body
        
        lines = body.splitlines()
        buffer = []

        for line in lines:
            line_stripped = line.strip()
            if line_stripped != "":
                buffer.append(line)

        return " ".join(buffer)
    
    def filter_unwanted_pages(self, documents: List[Document]) -> List[Document]:
        logger.info(f"Length of documents before filtering: {len(documents)}")
        cleaned_docs = []
        junk_keywords = ["contents", "contributors", "data sets and methods", "author", "table of contents", "credit", "editor", "cite", "citing", "references"]
        
        for doc in documents:
            if doc.metadata.get("page") == 0:
                continue
                
            page_text_lower = doc.page_content.lower()
            titles = [title.lower() for title in doc.metadata['headers']]

            text_has_junk = any(keyword in page_text_lower for keyword in junk_keywords)
            title_has_junk = any(keyword in title for keyword in junk_keywords for title in titles)
            
            if text_has_junk or title_has_junk:
                continue
                
            cleaned_docs.append(doc)

        logger.info(f"Length of documents after filtering: {len(cleaned_docs)}")
            
        return cleaned_docs
    
    def clean_document_text(self, doc: Document) -> Document:
        """
        Cleans the page_content of a LangChain Document by fixing
        PDF/Markdown extraction artifacts.
        
        - Removes Markdown image comments (e.g., ).
        - Removes entire lines that are just figure/source references.
        - Replaces non-breaking spaces (\xa0) with regular spaces.
        - Fixes hyphenation across newlines (e.g., "Govern-\nment" -> "Government").
        - Removes lines that *only* contain numbers (page numbers).
        - Removes "flattened" endnote numbers (e.g., "...climate. 19 This...")
        - Replaces single newlines (line breaks) with spaces (joins broken sentences).
        - Collapses multiple newlines (paragraph breaks) into a single newline.
        - Removes bracketed citation markers (e.g., [1], [22]).
        - Removes URLs.
        - Removes extra whitespace and stray backslashes.
        """
        text = doc.page_content
        text = text.replace('<!-- image -->', '')
        text = re.sub(r'^\s*(Figure|Source:).*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'https?://\S+', '', text)
        text = text.replace('\xa0', ' ')
        text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'\. \d{1,3}\b', '. ', text) # For "...2022. 22" or "...did. 5 Real-time..."
        text = re.sub(r',\s\d{1,3}\b', ', ', text) # For "...in 2021, 138 this..."
        text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
        text = re.sub(r'\n{2,}', '\n', text)
        text = re.sub(r'\[[0-9]{1,3}\]', '', text) 
        text = re.sub(r'\s+', ' ', text).strip()
        text = text.replace('\\', '')
        
        return Document(page_content=text, metadata=doc.metadata)

    def _load_and_split_markdown(self) -> List[Document]:
        path_to_md = pathlib.Path(f"data/drough/{self.file.stem}.md")
        md_text = path_to_md.read_text(encoding='utf-8')

        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
        ]
        md_splitter_obj = MarkdownHeaderTextSplitter(headers_to_split_on)
        splitted_md = md_splitter_obj.split_text(md_text)

        for doc in splitted_md:
            base_metadata = {'headers': []}
            for key in doc.metadata.keys():
                if "Header" in key:
                    base_metadata['headers'].append(doc.metadata[key].lower())
            doc.metadata = base_metadata

        return splitted_md
    
    def _extract_metadata_for_chunk(self, chunk: ChunkAndEmbedResponse, base_metadata: Dict) -> MyDocument | None:
        if len(chunk.text) < 50:
            return None
        try:
            # concatenate metadata/dicts together
            final_metadata = base_metadata.copy()
            
            # ask llm for metadata
            extracted_metadata = self.metadata_extractor.extract_metadata(prompt=DroughMetadata.DROUGH_METADATA_PROMPT.format(input_text=chunk.text), response_scheme=DroughMetadata.DroughMetadata)
            final_metadata = final_metadata | extracted_metadata
            for key, value in final_metadata.items():
                if isinstance(value, str):
                    final_metadata[key] = value.lower()
                elif isinstance(value, list):
                    lower_strings = []
                    for string_value in final_metadata[key]:
                        if isinstance(string_value, str):
                            lower_strings.append(string_value.lower())
                        else:
                            lower_strings.append(str(lower_strings).lower())
                    final_metadata[key] = lower_strings

            # ChromaDB cannot store lists, so we will convert lists to str, that will be divided by '|'
            if isinstance(self.db_repository, ChromaDbRepository):
                for key, value in final_metadata.items():
                    if isinstance(value, list):
                        valid_items = {str(item).strip() for item in value if str(item).strip()}
                        if valid_items:
                            final_metadata[key] = f"|{'|'.join(sorted(valid_items))}|"
                        else:
                            final_metadata[key] = 'None'

            return MyDocument(
                id=chunk.embed_text.uuid,
                text=chunk.text,
                embedding=chunk.embed_text.embedding,
                metadata=final_metadata
            )
    
        except IndentationError as ie:
            logger.error(f"Chat response: {extracted_metadata}")
            logger.info(f"Chunked text length {len(chunk.text)}: {chunk.text}")
            logger.error(f"Error during extract_metadata_for_chunk: {ie}")
            traceback.print_exc()
            sys.exit(1)
            self.state = ETLState.FAILED

    # TODO: Asi fajn napad, kdyz narazi na tabulku, tak to posle do llm a sumarizuje
    # def _summarize_tables_if_present(self, text: str, model: lms.LLM) -> str:
    #     pass

    def _process_document(self, doc: Document) -> List[MyDocument]:
        logger.info(f" Cleaning doc... {len(doc.page_content)}")
        doc = self.clean_document_text(doc)
        logger.info(f" Cleaned doc {len(doc.page_content)}")
        if len(doc.page_content) < 50:
            return []

        processed_chunks = []
        logger.info(f" Sending {len(doc.page_content)} length for chunk and embed")
        chunk_and_embed_result = self.embedding_service.chunk_and_embed(data=doc.page_content)
        logger.info(f" Chunk and embed result {len(chunk_and_embed_result)}")
        
        for chunk in chunk_and_embed_result:
            final_document = self._extract_metadata_for_chunk(chunk, doc.metadata)
            if final_document:
                processed_chunks.append(final_document)

        return processed_chunks

    def transform(self) -> None:

        try:
            # setup md
            raw_documents = self._load_and_split_markdown()
            documents = self.filter_unwanted_pages(raw_documents)
            #topics = set()

            for i, doc in enumerate(documents):
                logger.info(f"{i}. transforming document...")
                processed_docs = self._process_document(doc)
                if processed_docs:
                    self.documents.extend(processed_docs)
                    logger.info(f"{i}. transformed document (found {len(processed_docs)} chunks)")
                    logger.info(f"Current size of documents: {len(documents)}")
                    #topics.update(doc.metadata['topics'])

            # ONE POSSIBLE UPGRADE
            # logger.info("Optimalizing metadata for each document...")
            # for doc in self.documents:
            #     chat = lms.Chat(USER_METADATA_PERSONA)
            #     chat.add_user_message(METADATA_UPDATE_PROMPT.format(topics_list=topics, doc_sentence=doc.text, doc_metadata=doc.metadata['topics']))
            #     chat_response = model.respond(chat)

            #     print()
            #     logger.info(f"Chat response: {chat_response}")
            #     chat_response = eval(chat_response)
            #     logger.info(f"Topics original: {doc.metadata['topics']}")
            #     diff = set(chat_response) - set(doc.metadata['topics'])
            #     logger.info(f"Difference: {list(diff)}")
            #     print()
            logger.info(f"File: {self.file} was transformed!")
            logger.info(f"Number of documents: {len(self.documents)}")
            self.state = ETLState.TRANSFORMED
            return
        except Exception as e:
            logger.exception(f"Error during transform step: {e}")
            traceback.print_exc()
            self.state = ETLState.FAILED
            return


        


        

        
    
    