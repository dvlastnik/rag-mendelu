import json
import traceback
from typing import List, Tuple, Optional, Dict
import pandas as pd
import pathlib
import re
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
import sys

from etl.BaseEtl import BaseEtl, ETLState
from database.base.MyDocument import MyDocument, SparseVector
from text_embedding_api.TextEmbeddingService import ChunkAndEmbedResponse
from metadata_extractor.LLMMetadataExtractor import LLMMetadataExtractor
from metadata_extractor.models import DroughMetadata
from utils.logging_config import get_logger

logger = get_logger(__name__)

class DroughEtl(BaseEtl):
    def __init__(self, filepath, db_repositories, embedding_service, metadata_extractor: LLMMetadataExtractor):
        super().__init__(filepath, db_repositories, embedding_service)
        self.metadata_extractor = metadata_extractor

    def _sanitize_string(self, s: str) -> str:
        if not isinstance(s, str): 
            return str(s)
        return s.encode('utf-8', 'replace').decode('utf-8')

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
        text = text.replace('', '')
        text = re.sub(r'^\s*(Figure|Source:).*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'https?://\S+', '', text)
        text = text.replace('\xa0', ' ')
        text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'\. \d{1,3}(?=\s+[A-Z])', '.', text) 
        text = re.sub(r',\s\d{1,3}(?=\s)', ',', text)
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
    
    def _extract_section_metadata(self, text: str, existing_metadata: Dict) -> Dict:
        """
        Calls LLM to extract metadata for the whole section (document) 
        and merges it with existing header metadata.
        """
        try:
            extracted_metadata_obj = self.metadata_extractor.extract_metadata(
                prompt=DroughMetadata.DROUGH_METADATA_PROMPT.format(input_text=text[:4000]), # Truncate to be safe if section is huge
                response_scheme=DroughMetadata.DroughMetadata
            )
            extracted_metadata = extracted_metadata_obj.model_dump()
            
            final_metadata = existing_metadata.copy() | extracted_metadata

            processed_metadata = {}
            for key, value in final_metadata.items():
                if isinstance(value, str):
                    processed_metadata[key] = self._sanitize_string(value.lower())
                elif isinstance(value, list):
                    processed_metadata[key] = [
                        self._sanitize_string(str(item).lower()) 
                        for item in value
                    ]
                else:
                    processed_metadata[key] = value
            processed_metadata['source'] = self.file.stem
            
            return processed_metadata

        except Exception as e:
            logger.error(f"LLM Metadata extraction failed: {e}")
            return existing_metadata
    
    def _create_document_from_chunk(self, chunk: ChunkAndEmbedResponse, full_metadata: Dict) -> MyDocument | None:
        if len(chunk.text) < 50:
            return None
        
        try:
            chunk_metadata = full_metadata.copy()

            sanitized_text = self._sanitize_string(chunk.text)
            chunk_metadata['text'] = chunk.text

            sparse_vec = None
            if chunk.sparse_embedding:
                sparse_vec = SparseVector(chunk.sparse_embedding.indices, chunk.sparse_embedding.values)

            return MyDocument(
                id=chunk.embed_text.uuid,
                text=sanitized_text,
                embedding=chunk.embed_text.embedding,
                sparse_embedding=sparse_vec,
                metadata=chunk_metadata
            )
    
        except Exception as e:
            logger.error(f"Error creating MyDocument: {e}")
            traceback.print_exc()
            return None

    def _process_document(self, text: str, metadata: Dict) -> List[MyDocument]:
        if len(text) < 50:
            return []

        processed_chunks = []
        logger.debug(f" Sending {len(text)} chars to ChunkAndEmbed API")
        chunk_and_embed_result = self.embedding_service.chunk_and_embed(data=text)
        
        for chunk in chunk_and_embed_result:
            chunk.text = chunk.text.lstrip(',')
            
            final_document = self._create_document_from_chunk(chunk, metadata)
            if final_document:
                processed_chunks.append(final_document)

        return processed_chunks

    def transform(self) -> None:
        try:
            raw_documents = self._load_and_split_markdown()
            documents = self.filter_unwanted_pages(raw_documents)

            total_docs = len(documents)
            logger.info(f"Starting transform for {total_docs} sections...")

            for i, doc in enumerate(documents):
                index = i + 1
                
                cleaned_doc = self.clean_document_text(doc)
                if len(cleaned_doc.page_content) < 50: 
                    continue

                logger.info(f"[{index}/{total_docs}] Processing section...")

                enhanced_metadata = self._extract_section_metadata(
                    text=cleaned_doc.page_content, 
                    existing_metadata=cleaned_doc.metadata
                )

                processed_docs = self._process_document(cleaned_doc.page_content, enhanced_metadata)
                
                if processed_docs:
                    self.documents.extend(processed_docs)
                    logger.debug(f" -> Generated {len(processed_docs)} chunks.")

            logger.info(f"File: {self.file} transformed! Total chunks: {len(self.documents)}")
            self.state = ETLState.TRANSFORMED
            return

        except Exception as e:
            logger.exception(f"Error during transform step: {e}")
            traceback.print_exc()
            self.state = ETLState.FAILED
            return