import traceback
import httpx
from typing import List, Tuple, Optional, Dict
from collections import Counter
import pathlib
import re
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from etl.BaseEtl import BaseEtl, ETLState
from database.base.MyDocument import MyDocument, SparseVector
from text_embedding_api.TextEmbeddingService import ChunkAndEmbedResponse
from utils.logging_config import get_logger

logger = get_logger(__name__)

MAX_SAFE_CHUNK_SIZE = 2000

class DroughEtl(BaseEtl):
    def __init__(self, filepath, db_repositories, embedding_service, metadata_extractor):
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
        """
        text = doc.page_content
        text = re.sub(r'', '', text, flags=re.DOTALL)
        text = re.sub(r'<!-- image -->', '', text, flags=re.IGNORECASE)
        text = text.replace('\x00', '')
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
    
    def _merge_metadata_dicts(self, results: List[Dict]) -> Dict:
        """
        Helper to merge multiple extraction results.
        - Lists (cities): Union of all found items.
        - Singletons (year): Most frequent value (Voting) OR collect all unique.
        """
        merged = {}
        
        if not results: return {}
        keys = results[0].keys()

        for key in keys:
            values = [r.get(key) for r in results if r.get(key) not in [None, [], ""]]
            
            if not values:
                merged[key] = None
                continue

            first_val = values[0]
            if isinstance(first_val, list):
                combined = []
                for v_list in values:
                    combined.extend(v_list)
                merged[key] = list(set(combined))
                
            else:
                most_common = Counter(values).most_common(1)[0][0]
                merged[key] = most_common

        return merged
    
    def _extract_section_metadata(self, text: str, existing_metadata: Dict) -> Dict:
        """
        Extracts metadata by segmenting long text to fit smaller model context,
        then aggregates the results.
        """
        try:
            segment_size = 2000
            overlap = 100
            
            if len(text) <= segment_size:
                segments = [text]
            else:
                segments = []
                start = 0
                while start < len(text):
                    end = min(start + segment_size, len(text))
                    segments.append(text[start:end])
                    start += (segment_size - overlap)

            extracted_objects = []
            for i, segment in enumerate(segments):
                try:
                    metadata_from_agent = self.metadata_extractor.invoke({'text_chunk': segment})
                    appending_metadata = {
                        'years': metadata_from_agent['clean_data']['years'],
                        'locations': metadata_from_agent['clean_data']['locations']
                    }
                    extracted_objects.append(appending_metadata)

                except (TimeoutError, httpx.ReadTimeout, httpx.ConnectTimeout):
                    logger.error(f"   [Segment {i+1}] OLLAMA TIMEOUT. Skipping segment.")
                    break
                except Exception as e:
                    traceback.print_exc()
                    logger.error(f"   [Segment {i+1}] General Error: {e}")
                    continue

            if not extracted_objects:
                logger.warning("No metadata extracted from any segment.")
                return existing_metadata

            aggregated_data = self._merge_metadata_dicts(extracted_objects)
            final_metadata = existing_metadata.copy() | aggregated_data
            
            processed_metadata = {}
            for key, value in final_metadata.items():
                if value is None:
                    continue
                
                if isinstance(value, str):
                    processed_metadata[key] = self._sanitize_string(value.lower())
                elif isinstance(value, list):
                    processed_metadata[key] = sorted(list(set([
                        self._sanitize_string(item.lower()) if isinstance(item, str) else item
                        for item in value
                    ])))
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

    def _process_document(self, text: str, base_metadata: Dict) -> List[MyDocument]:
        if len(text) < 50:
            return []

        segments_to_process = [MyDocument(id='null', text=text, metadata=base_metadata.copy())]
        if len(text) > MAX_SAFE_CHUNK_SIZE:
            logger.warning(f"Text too large ({len(text)} chars). Pre-splitting locally.")
            pre_splitter = RecursiveCharacterTextSplitter(
                chunk_size=MAX_SAFE_CHUNK_SIZE,
                chunk_overlap=200,
                separators=["\n\n", "\n", " ", ""]
            )
            raw_texts = pre_splitter.split_text(text)
            
            segments_to_process: List[MyDocument] = []
            for t in raw_texts:
                segments_to_process.append(MyDocument(id='null',text=t, metadata=base_metadata.copy()))
            
            logger.info(f' - Smaller chunks len: {len(segments_to_process)}')

        processed_final_docs = []
        for i, segment in enumerate(segments_to_process):
            current_text = segment.text
            current_base_metadata = segment.metadata

            enhanced_metadata = self._extract_section_metadata(
                text=current_text,
                existing_metadata=current_base_metadata
            )
            
            try:
                if len(segments_to_process) > 1:
                    logger.info(f' - [{i+1}/{len(segments_to_process)}] Processing segment...')
                
                response_chunks = self.embedding_service.chunk_and_embed(data=current_text)
                
                if not response_chunks:
                    logger.warning(f' - [{i+1}/{len(segments_to_process)}] Response was empty')
                    continue

                for semantic_chunk in response_chunks:
                    semantic_chunk.text = semantic_chunk.text.lstrip(',')
                    
                    final_document = self._create_document_from_chunk(
                        semantic_chunk,
                        enhanced_metadata
                    )
                    
                    if final_document:
                        processed_final_docs.append(final_document)

            except Exception as e:
                logger.error(f"Failed to process segment {i}: {e}")

        return processed_final_docs

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

                logger.info(f"[{index}/{total_docs}] Processing section of length: {len(cleaned_doc.page_content)}...")

                cleaned_metadata = cleaned_doc.metadata.copy()
                cleaned_metadata['parent_text'] = cleaned_doc.page_content

                processed_docs = self._process_document(cleaned_doc.page_content, cleaned_metadata)
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