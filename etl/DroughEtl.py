import traceback
from typing import List, Optional, Dict
from collections import Counter
import pathlib
import re
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from etl.BaseEtl import BaseEtl, ETLState
from etl.table_extractor import TableProcessor
from database.base.MyDocument import MyDocument, SparseVector
from text_embedding import EmbeddingResponse
from semantic_chunking.sentence_similarity import SentenceSimilarity
from semantic_chunking.similiar_sentence_splitter import SimilarSentenceSplitter
from utils.Utils import Utils
from utils.logging_config import get_logger

logger = get_logger(__name__)

_RE_IMAGE_COMMENT = re.compile(r'<!-- image -->', re.IGNORECASE)
_RE_FIGURE_SOURCE = re.compile(r'^\s*(Figure|Source:).*$', re.MULTILINE)
_RE_URL = re.compile(r'https?://\S+')
_RE_HYPHENATION = re.compile(r'(\w)-\n(\w)')
_RE_STANDALONE_NUMBER = re.compile(r'^\s*\d+\s*$', re.MULTILINE)
_RE_TRAILING_PAGE_NUM = re.compile(r'\. \d{1,3}(?=\s+[A-Z])')
_RE_COMMA_PAGE_NUM = re.compile(r',\s\d{1,3}(?=\s)')
_RE_SINGLE_NEWLINE = re.compile(r'(?<!\n)\n(?!\n)')
_RE_MULTI_NEWLINE = re.compile(r'\n{2,}')
_RE_CITATION = re.compile(r'\[[0-9]{1,3}\]')
_RE_MULTI_SPACE = re.compile(r'\s+')

MIN_CHUNK_SIZE = 50
METADATA_EXTRACTION_CONTEXT_SIZE = 3000
MAX_CHUNK_SIZE_FOR_SEMANTIC_SEARCH = 3000

class DroughtEtl(BaseEtl):
    OUTPUT_FOLDER = "data/drough"

    """
    ETL pipeline for processing climate/drought-related markdown documents.
    
    Handles markdown ingestion, text cleaning, metadata extraction via LLM,
    and embedding generation for vector database storage.
    
    Note: Currently optimized for drought/climate reports but designed to be
    extensible for other document types in the future.
    """
    JUNK_TITLE_KEYWORDS = [
        "contents",
        "highlights",
        "table of contents",
        "contributors", 
        "data sets and methods",
        "author",
        "credits",
        "editors",
        "citing this report",
        "references",
        "acknowledgements",
        "executive summary"
    ]
    
    def __init__(
        self, 
        filepath: pathlib.Path, 
        db_repositories, 
        embedding_service, 
        metadata_extractor,
        use_semantic: bool = True
    ) -> None:
        """
        Initialize the DroughtEtl pipeline.
        
        Args:
            filepath: Path to the markdown file to process.
            db_repositories: Database repository instances for storage.
            embedding_service: Service for generating text embeddings.
            metadata_extractor: LLM-based metadata extraction graph/chain.
            use_semantic: Whether to use semantic chunking (for text <= 2000 chars).
        """
        super().__init__(filepath, db_repositories, embedding_service)
        self.metadata_extractor = metadata_extractor
        self.table_processor = TableProcessor()
        self.use_semantic = use_semantic

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        sentence_similarity_model = SentenceSimilarity(embedding_service=embedding_service)
        self.semantic_splitter = SimilarSentenceSplitter(similarity_model=sentence_similarity_model)

    def _sanitize_string(self, s: Optional[str]) -> str:
        """
        Sanitize a string by removing null bytes and invalid unicode characters.
        
        Args:
            s: Input string (can be None or non-string type).
            
        Returns:
            Cleaned string with invalid characters removed.
        """
        if s is None:
            return ""
        if not isinstance(s, str):
            s = str(s)
        return s.replace('\x00', '').replace('\ufffd', '').strip()

    def get_file_path(self, only_folder: bool = False, chunk_index: Optional[int] = None) -> pathlib.Path:
        """
        Generate output file path for processed chunks.
        
        Args:
            only_folder: If True, return only the folder path.
            chunk_index: Optional chunk index for individual chunk files.
            
        Returns:
            Path object for the output location.
        """
        folder_prefix = "data/ignore_"

        if only_folder:
            return pathlib.Path(f"{folder_prefix}{self.file.stem}")
        if chunk_index is not None:
            return pathlib.Path(f"{folder_prefix}{self.file.stem}/{self.file.stem}_{chunk_index}.md")
        return pathlib.Path(f"{folder_prefix}{self.file.stem}/{self.file.stem}.md")
    
    def _filter_unwanted_sections(self, documents: List[Document]) -> List[Document]:
        """
        Filter out non-content sections like table of contents, contributors, etc.
        
        Uses title-based filtering only to avoid false positives from body text
        containing keywords like "contents" (e.g., "ocean heat contents").
        
        Args:
            documents: List of Document objects from markdown splitting.
            
        Returns:
            Filtered list with junk sections removed.
        """
        logger.info(f"Sections before filtering: {len(documents)}")
        cleaned_docs = []
        
        for doc in documents:
            headers = doc.metadata.get('headers', [])
            header_text = ' '.join(headers).lower()
            
            is_junk_section = any(
                keyword in header_text 
                for keyword in self.JUNK_TITLE_KEYWORDS
            )

            if not is_junk_section:
                is_junk_section = '...............' in doc.page_content
            
            if is_junk_section:
                logger.debug(f"Filtered section with headers: {headers}")
                continue
                
            cleaned_docs.append(doc)

        logger.info(f"Sections after filtering: {len(cleaned_docs)}")
        return cleaned_docs
    
    def _clean_document_text(self, doc: Document) -> Document:
        """
        Clean the page_content of a Document by removing extraction artifacts.
        
        Handles:
        - HTML/Markdown comments and image placeholders
        - URLs and citation markers
        - Hyphenated line breaks from PDF extraction
        - Standalone page numbers
        - Excessive whitespace
        
        Args:
            doc: LangChain Document with raw extracted text.
            
        Returns:
            New Document with cleaned page_content (metadata preserved).
        """
        text = doc.page_content
        
        text = _RE_IMAGE_COMMENT.sub('', text)
        text = text.replace('\x00', '').replace('\xa0', ' ')
        text = _RE_HYPHENATION.sub(r'\1\2', text)
        text = _RE_FIGURE_SOURCE.sub('', text)
        text = _RE_URL.sub('', text)
        text = _RE_STANDALONE_NUMBER.sub('', text)
        text = _RE_TRAILING_PAGE_NUM.sub('.', text)
        text = _RE_COMMA_PAGE_NUM.sub(',', text)
        text = _RE_SINGLE_NEWLINE.sub(' ', text)
        text = _RE_MULTI_NEWLINE.sub('\n', text)
        text = _RE_CITATION.sub('', text)
        text = _RE_MULTI_SPACE.sub(' ', text).strip()
        text = text.replace('\\', '')
        
        return Document(page_content=text, metadata=doc.metadata)

    def _load_and_split_markdown(self) -> List[Document]:
        """
        Load markdown file, extract tables, and split by headers into Document sections.
        
        Tables are extracted first from the full document, then removed from the text
        before splitting by headers. Table Documents are added to the result list.
        
        Each text Document contains:
        - page_content: The text under that header
        - metadata['headers']: List of header texts (hierarchical)
        - metadata['header_path']: Full header hierarchy as string
        - metadata['is_table']: False
        
        Each table Document contains:
        - page_content: The table summary text
        - metadata['is_table']: True
        - metadata['table_rows'], ['table_columns'], ['table_headers'], etc.
        
        Returns:
            List of Documents (both text sections and tables).
            
        Raises:
            FileNotFoundError: If the markdown file doesn't exist.
        """
        output_md_file = Utils.get_output_path(self.file, 'data/drough')
        md_text = output_md_file.read_text(encoding='utf-8')

        logger.info(f"Extracting tables from '{self.file.name}' ({len(md_text):,} chars)...")
        text_without_tables, table_documents = self.table_processor.process_document(
            markdown_text=md_text,
            base_metadata={'source': self.file.stem}
        )
        
        if table_documents:
            logger.info(f" - Extracted {len(table_documents)} tables from document")
        else:
            logger.info(f" - No tables found in '{self.file.name}'")

        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
        ]
        md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
        split_docs = md_splitter.split_text(text_without_tables)

        for doc in split_docs:
            headers = []
            for key in sorted(doc.metadata.keys()):
                if "Header" in key:
                    headers.append(doc.metadata[key].lower())
            doc.metadata = {
                'headers': headers,
                'header_path': ' > '.join(headers) if headers else '',
                'is_table': False
            }

        for table_doc in table_documents:
            table_metadata = table_doc['metadata'].copy()
            table_metadata['is_table'] = True
            
            doc = Document(
                page_content=table_doc['text'],
                metadata=table_metadata
            )
            split_docs.append(doc)

        return split_docs
    
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
        Extracts metadata from the provided context (Section Header/Intro).
        Performs ONE robust LLM call instead of multiple flaky ones.
        """
        try:
            metadata_from_agent = self.metadata_extractor.invoke({'text_chunk': text})
            clean_data = metadata_from_agent.get('clean_data', {})
            new_years = clean_data.get('years', [])
            new_locations = clean_data.get('locations', [])
            new_entities = clean_data.get('entities', [])
            has_numerical_data = clean_data.get('has_numerical_data', False)
            
            final_metadata = existing_metadata.copy()
            
            if 'years' in final_metadata and isinstance(final_metadata['years'], list):
                final_metadata['years'].extend(new_years)
            else:
                final_metadata['years'] = new_years

            if 'locations' in final_metadata and isinstance(final_metadata['locations'], list):
                final_metadata['locations'].extend(new_locations)
            else:
                final_metadata['locations'] = new_locations
            
            if 'entities' in final_metadata and isinstance(final_metadata['entities'], list):
                final_metadata['entities'].extend(new_entities)
            else:
                final_metadata['entities'] = new_entities
            
            final_metadata['has_numerical_data'] = final_metadata.get('has_numerical_data', False) or has_numerical_data

            processed_metadata = {}
            for key, value in final_metadata.items():
                if value is None: 
                    continue
                
                if isinstance(value, str):
                    processed_metadata[key] = self._sanitize_string(value.lower())
                elif isinstance(value, list):
                    if key == 'years':
                        processed_metadata[key] = sorted(list(set([int(item) for item in value if item])))
                    else:
                        processed_metadata[key] = sorted(list(set([
                            self._sanitize_string(str(item).lower()) for item in value if item
                        ])))
                elif isinstance(value, bool):
                    processed_metadata[key] = value
                else:
                    processed_metadata[key] = value
            
            processed_metadata['source'] = self.file.stem
            return processed_metadata

        except Exception as e:
            logger.error(f"LLM Metadata extraction failed: {e}")
            return existing_metadata
    
    def _create_document_from_chunk(self, embedding_data: EmbeddingResponse, text: str, full_metadata: Dict) -> MyDocument | None:
        """
        Creates a MyDocument object by combining the Text (from splitter) 
        and the Vectors (from EmbeddingResponse).
        """
        if not text or len(text) < 50:
            return None
        
        try:
            chunk_metadata = full_metadata.copy()
            sparse_vec = None
            if embedding_data.sparse:
                sparse_vec = SparseVector(
                    indices=embedding_data.sparse.indices, 
                    values=embedding_data.sparse.values
                )

            return MyDocument(
                id=embedding_data.uuid,
                text=text,
                embedding=embedding_data.embedding,
                sparse_embedding=sparse_vec,
                metadata=chunk_metadata
            )
    
        except Exception as e:
            logger.error(f"Error creating MyDocument: {e}")
            traceback.print_exc()
            return None

    def _process_document(self, text: str, base_metadata: Dict) -> List[MyDocument]:
        """
        Process a single document section: extract metadata, chunk, and embed.
        
        When use_semantic is True:
        - Text <= 2000 chars: Send directly to semantic chunk_and_embed API
        - Text > 2000 chars: Pre-split with recursive splitter, then send each chunk to semantic API
        
        When use_semantic is False:
        - Use recursive splitter + get_embedding_with_uuid (traditional approach)
        
        Args:
            text: The cleaned section text to process.
            base_metadata: Metadata from the document (headers, parent_text, etc.)
            
        Returns:
            List of MyDocument objects ready for database insertion.
        """
        if not text or len(text) < MIN_CHUNK_SIZE:
            return []

        processed_final_docs = []
        try:
            # extraction_context = text[:METADATA_EXTRACTION_CONTEXT_SIZE]
            # enhanced_metadata = self._extract_section_metadata(
            #     text=extraction_context,
            #     existing_metadata=base_metadata
            # )

            if self.use_semantic:
                logger.info(f"Using semantic splitter for section ({len(text)} chars)...")
                raw_text_chunks = self.semantic_splitter.split_text(text)
                    
            else:
                logger.info(f"Using recursive splitter for section ({len(text)} chars)...")
                raw_text_chunks = self.splitter.split_text(text)
                
            if not raw_text_chunks:
                logger.warning("Splitter returned no chunks.")
                return []

            valid_chunks = [
                self._sanitize_string(chunk) 
                for chunk in raw_text_chunks 
                if len(self._sanitize_string(chunk)) >= MIN_CHUNK_SIZE
            ]

            if not valid_chunks:
                return []

            logger.info(f"Split section into {len(valid_chunks)} valid chunks. Sending batch to Embedding service...")
            embedding_responses: List[EmbeddingResponse] = self.embedding_service.get_embedding_with_uuid(
                data=valid_chunks, 
                chunk_size=32
            )

            if len(embedding_responses) != len(valid_chunks):
                logger.error(f"CRITICAL MISMATCH: Sent {len(valid_chunks)} chunks, got {len(embedding_responses)} embeddings.")
                return []

            for i, response_obj in enumerate(embedding_responses):
                logger.info(' - Extracting metadata...')
                enhanced_metadata = self._extract_section_metadata(
                    text=valid_chunks[i],
                    existing_metadata=base_metadata
                )
                logger.info('-------------------------')

                chunk_metadata = enhanced_metadata.copy()
                chunk_metadata['text'] = valid_chunks[i]
                chunk_metadata['chunking_method'] = 'recursive'

                final_document = self._create_document_from_chunk(
                    embedding_data=response_obj,
                    text=valid_chunks[i],
                    full_metadata=chunk_metadata
                )
                
                if final_document:
                    processed_final_docs.append(final_document)

        except Exception as e:
            logger.error(f"Failed to process document section: {e}")
            traceback.print_exc()
            
        return processed_final_docs

    def transform(self) -> None:
        """
        Main transformation pipeline: load, filter, clean, extract metadata, embed.
        
        Processes the markdown file through:
        1. Load and split by headers (tables already extracted and included)
        2. Filter out non-content sections
        3. For tables: embed directly without chunking
        4. For text: clean, extract metadata via LLM, chunk, and embed
        5. Create MyDocument objects for storage
        
        Sets self.state to TRANSFORMED on success, FAILED on error.
        Results are stored in self.documents.
        """
        try:
            raw_documents = self._load_and_split_markdown()
            documents = self._filter_unwanted_sections(raw_documents)
            total_docs = len(documents)
            logger.info(f"Processing {total_docs} text sections from '{self.file.name}'...")

            for i, doc in enumerate(documents):
                index = i + 1
                
                cleaned_doc = self._clean_document_text(doc)
                if len(cleaned_doc.page_content) < MIN_CHUNK_SIZE:
                    logger.debug(f"[{index}/{total_docs}] Skipping short section ({len(cleaned_doc.page_content)} chars)")
                    continue

                logger.info(f"[{index}/{total_docs}] Processing section: {cleaned_doc.metadata.get('header_path', 'No headers')} ({len(cleaned_doc.page_content)} chars)")

                section_metadata = cleaned_doc.metadata.copy()
                section_metadata['parent_text'] = cleaned_doc.page_content

                processed_docs = self._process_document(cleaned_doc.page_content, section_metadata)
                if processed_docs:
                    self.documents.extend(processed_docs)
                    logger.debug(f" -> Generated {len(processed_docs)} chunks")

            logger.info(
                f"✓ Transform complete for '{self.file.name}': "
                f"{len(self.documents)} total documents "
            )
            self.state = ETLState.TRANSFORMED

        except FileNotFoundError as e:
            logger.error(f"File not found: {self.file} - {e}")
            self.state = ETLState.FAILED
        except Exception as e:
            logger.exception(f"Error during transform step: {e}")
            self.state = ETLState.FAILED