import gc
import re
import traceback
import pathlib
from typing import List, Dict, Optional
import pandas as pd
from tqdm import tqdm
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

from etl.base_etl import BaseEtl, ETLState
from etl.table_extractor import TableProcessor
from database.base.my_document import MyDocument, SparseVector
from database.base.base_db_repository import BaseDbRepository
from database.duck_db_repository import DuckDbRepository
from text_embedding.text_embedding_service import TextEmbeddingService
from text_embedding import EmbeddingResponse
from semantic_chunking.sentence_similarity import SentenceSimilarity
from semantic_chunking.similiar_sentence_splitter import SimilarSentenceSplitter
from utils.logging_config import get_logger

logger = get_logger(__name__)

MIN_CHUNK_SIZE = 50

_RE_HTML_COMMENT = re.compile(r'<!--.*?-->', re.DOTALL)
_RE_URL = re.compile(r'https?://\S+')
_RE_HYPHENATION = re.compile(r'(\w)-\n(\w)')
_RE_SINGLE_NEWLINE = re.compile(r'(?<!\n)\n(?!\n)')
_RE_MULTI_NEWLINE = re.compile(r'\n{3,}')
_RE_MULTI_SPACE = re.compile(r'[ \t]+')

_RE_MD_BOLD = re.compile(r'\*\*(.+?)\*\*|__(.+?)__', re.DOTALL)
_RE_MD_ITALIC = re.compile(r'\*(.+?)\*', re.DOTALL)
_RE_MD_CODE = re.compile(r'`(.+?)`', re.DOTALL)
_RE_MD_LINK = re.compile(r'\[([^\]]+)\]\([^)]+\)')
_RE_MD_BLOCKQUOTE = re.compile(r'^>\s?', re.MULTILINE)
_RE_MD_HRULE = re.compile(r'^[-=*_]{3,}\s*$', re.MULTILINE)


class GeneralEtl(BaseEtl):
    """
    General-purpose ETL pipeline for any supported file type.

    Supports: .pdf, .docx, .pptx (via Docling), .md, .txt (native text),
    .csv, .xlsx (tabular → markdown).

    Pipeline: convert to Markdown → extract tables → split by headers →
    semantic/recursive chunk → dense+sparse embed → Qdrant.
    """

    OUTPUT_FOLDER = "data/general"
    SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.pptx', '.md', '.txt', '.csv', '.xlsx'}

    def __init__(
        self,
        filepath: str,
        db_repository: BaseDbRepository,
        embedding_service: TextEmbeddingService,
        sql_db_repo: DuckDbRepository
    ) -> None:
        super().__init__(filepath, db_repository, embedding_service)
        self.sql_db_repo = sql_db_repo
        self.table_processor = TableProcessor()

        sentence_similarity = SentenceSimilarity(embedding_service=embedding_service)
        self.semantic_splitter = SimilarSentenceSplitter(similarity_model=sentence_similarity)

    def get_file_path(self, only_folder: bool = False, chunk_index: Optional[int] = None) -> pathlib.Path:
        folder = pathlib.Path(self.OUTPUT_FOLDER)
        if only_folder:
            return folder
        return folder / f"{self.file.stem}.md"

    def transform(self) -> None:
        try:
            if self.file.suffix.lower() in {'.csv', '.xlsx'}:
                self._process_tabular()
                if self.state != ETLState.FAILED:
                    self.state = ETLState.TRANSFORMED
                return

            md_text = self._read_markdown()
            if not md_text:
                logger.error("No markdown content to process.")
                self.state = ETLState.FAILED
                return

            base_metadata = {
                'source': self.file.stem,
                'file_type': self.file.suffix.lower().lstrip('.'),
            }

            text_without_tables, table_documents = self.table_processor.process_document(
                markdown_text=md_text,
                base_metadata=base_metadata,
            )

            split_docs = self._split_by_headers(text_without_tables)
            del md_text, text_without_tables  # free large strings before embedding loop

            total = len(split_docs)
            logger.info(f"Processing {total} sections from '{self.file.name}'...")

            total_inserted = 0
            for i, doc in enumerate(tqdm(split_docs, desc=f"  {self.file.name}", unit="section", leave=False), 1):
                cleaned = self._clean_text(doc.page_content)
                if not cleaned:
                    continue

                section_metadata = {**base_metadata, **doc.metadata}
                logger.info(f"[{i}/{total}] {section_metadata.get('header_path', '(no headers)')} ({len(cleaned)} chars)")

                processed = self._process_section(cleaned, section_metadata)
                if processed:
                    insert_result = self.db_repository.insert(processed)
                    if not insert_result.success:
                        logger.error(f"Insert failed for section: {insert_result.message}")
                        self.state = ETLState.FAILED
                        return
                    total_inserted += len(processed)
                    gc.collect()

            table_docs_batch: List[MyDocument] = []
            for table_doc in table_documents:
                table_text = table_doc['text']
                if not table_text:
                    continue

                table_meta = {**base_metadata, **table_doc['metadata']}
                responses = self.embedding_service.get_embedding_with_uuid(
                    data=[table_text], chunk_size=None
                )
                if responses:
                    r = responses[0]
                    sparse_vec = SparseVector(indices=r.sparse.indices, values=r.sparse.values) if r.sparse else None
                    table_docs_batch.append(MyDocument(
                        id=r.uuid,
                        text=table_text,
                        embedding=r.embedding,
                        sparse_embedding=sparse_vec,
                        metadata=table_meta,
                    ))

            if table_docs_batch:
                insert_result = self.db_repository.insert(table_docs_batch)
                if not insert_result.success:
                    logger.error(f"Insert failed for tables: {insert_result.message}")
                    self.state = ETLState.FAILED
                    return
                total_inserted += len(table_docs_batch)

            if self.sql_db_repo is not None and table_documents:
                from collections import defaultdict
                _INTERNAL = {'is_table', 'source', 'file_type', 'table_index'}
                by_table: dict = defaultdict(list)
                for doc in table_documents:
                    idx = doc['metadata'].get('table_index', 0)
                    row = {k: v for k, v in doc['metadata'].items() if k not in _INTERNAL}
                    if row:
                        by_table[idx].append(row)

                for idx, rows in sorted(by_table.items()):
                    if not rows:
                        continue
                    sample_keys = list(rows[0].keys())
                    bad = sum(1 for k in sample_keys if len(k) <= 1)
                    if sample_keys and bad / len(sample_keys) > 0.6:
                        logger.warning(
                            f"Skipping DuckDB registration for '{self.file.stem}_table{idx}': "
                            f"{bad}/{len(sample_keys)} column keys are empty/single-char after sanitization."
                        )
                        continue
                    df = pd.DataFrame(rows)
                    table_name = f"{self.file.stem}_table{idx}"
                    self.sql_db_repo.register_dataframe(table_name, df)

            print(f"    {total_inserted} chunks inserted")
            logger.info(f"Transform complete: {total_inserted} documents from '{self.file.name}'")
            self.state = ETLState.TRANSFORMED

        except FileNotFoundError as e:
            logger.error(f"File not found during transform: {e}")
            self.state = ETLState.FAILED
        except Exception as e:
            logger.exception(f"Transform failed: {e}")
            self.state = ETLState.FAILED

    def _read_markdown(self) -> str:
        """Read markdown content from the converted output file on disk."""
        md_path = self.get_file_path()
        if not md_path.exists():
            raise FileNotFoundError(f"Converted markdown not found: {md_path}")
        return md_path.read_text(encoding='utf-8')

    _TABULAR_BATCH_SIZE = 500

    def _process_tabular(self) -> None:
        """Process CSV/XLSX as one MyDocument per row, streaming inserts in batches.

        Each row is serialised to a pipe-delimited text string and embedded.
        Every column value is stored as a typed metadata key so Qdrant can
        filter on exact values or numeric ranges.  The full text is also stored
        in metadata under the ``text`` key for easy inspection.

        Rows are processed in batches of _TABULAR_BATCH_SIZE to keep memory
        bounded regardless of file size.
        """
        if self.df is None or self.df.empty:
            logger.error("DataFrame is empty or None — cannot process tabular data.")
            self.state = ETLState.FAILED
            return

        columns = list(self.df.columns)
        file_type = self.file.suffix.lower().lstrip('.')

        base_metadata: Dict = {
            'source': self.file.stem,
            'file_type': file_type,
            'is_table': False,
        }

        total_rows = len(self.df)
        total_inserted = 0

        with tqdm(total=total_rows, desc=f"  {self.file.name}", unit="row", leave=False) as pbar:
            for batch_start in range(0, total_rows, self._TABULAR_BATCH_SIZE):
                batch_df = self.df.iloc[batch_start:batch_start + self._TABULAR_BATCH_SIZE]

                texts: List[str] = []
                row_metadatas: List[Dict] = []

                for row_index, row in batch_df.iterrows():
                    parts: List[str] = []
                    row_meta: Dict = {}

                    for col in columns:
                        val = row[col]
                        try:
                            if pd.isna(val):
                                continue
                        except (TypeError, ValueError):
                            pass

                        parts.append(f"{col}: {val}")
                        row_meta[col] = self._coerce_value(val)

                    pbar.update(1)
                    if not parts:
                        continue

                    text = " | ".join(parts)
                    texts.append(text)
                    row_metadatas.append({
                        **row_meta,
                        **base_metadata,
                        'row_index': int(row_index),
                        'text': text,
                    })

                if not texts:
                    continue

                logger.info(f"Embedding batch rows {batch_start}–{batch_start + len(texts) - 1} from '{self.file.name}'...")
                responses = self.embedding_service.get_embedding_with_uuid(data=texts, chunk_size=32)

                if len(responses) != len(texts):
                    logger.error(
                        f"Embedding count mismatch: {len(texts)} rows, {len(responses)} responses"
                    )
                    self.state = ETLState.FAILED
                    return

                batch_docs: List[MyDocument] = []
                for text, meta, r in zip(texts, row_metadatas, responses):
                    sparse_vec = SparseVector(indices=r.sparse.indices, values=r.sparse.values) if r.sparse else None
                    batch_docs.append(MyDocument(
                        id=r.uuid,
                        text=text,
                        embedding=r.embedding,
                        sparse_embedding=sparse_vec,
                        metadata=meta,
                    ))

                insert_result = self.db_repository.insert(batch_docs)
                if not insert_result.success:
                    logger.error(f"Insert failed for tabular batch: {insert_result.message}")
                    self.state = ETLState.FAILED
                    return

                total_inserted += len(batch_docs)
                gc.collect()

        if total_inserted == 0:
            logger.warning(f"No valid rows found in '{self.file.name}'")
            return

        logger.info(f"Tabular transform complete: {total_inserted} documents from '{self.file.name}'")

        if self.sql_db_repo is not None:
            suffix = self.file.suffix.lower()
            if suffix == '.csv':
                self.sql_db_repo.register_csv(self.file.stem, str(self.file.resolve()))
            elif suffix == '.xlsx':
                self.sql_db_repo.register_xlsx(self.file.stem, str(self.file.resolve()))

    @staticmethod
    def _coerce_value(val) -> int | float | bool | str:
        """Convert a pandas cell value to a JSON-serialisable Python primitive."""
        if isinstance(val, bool):
            return bool(val)
        if isinstance(val, (int,)):
            return int(val)
        if isinstance(val, float):
            return int(val) if val == int(val) else float(val)
        return str(val)

    def _split_by_headers(self, md_text: str) -> List[Document]:
        """Split markdown by H1–H4 headers and normalise metadata."""
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
        ]
        splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
        docs = splitter.split_text(md_text)

        for doc in docs:
            headers = [
                doc.metadata[k].lower()
                for k in sorted(doc.metadata)
                if "Header" in k
            ]
            doc.metadata = {
                'headers': headers,
                'header_path': ' > '.join(headers) if headers else '',
                'is_table': False,
            }
        return docs

    def _clean_text(self, text: str) -> str:
        """Generic text cleaning — strips markdown syntax and normalises whitespace."""
        text = _RE_MD_HRULE.sub('', text)
        text = _RE_MD_BLOCKQUOTE.sub('', text)
        text = _RE_MD_LINK.sub(r'\1', text)
        text = _RE_MD_CODE.sub(r'\1', text)
        text = _RE_MD_BOLD.sub(lambda m: m.group(1) or m.group(2), text)
        text = _RE_MD_ITALIC.sub(r'\1', text)

        text = _RE_HTML_COMMENT.sub('', text)
        text = text.replace('\x00', '').replace('\xa0', ' ')
        text = _RE_HYPHENATION.sub(r'\1\2', text)
        text = _RE_URL.sub('', text)
        text = _RE_SINGLE_NEWLINE.sub(' ', text)
        text = _RE_MULTI_NEWLINE.sub('\n\n', text)
        text = _RE_MULTI_SPACE.sub(' ', text)
        return text.strip()

    def _process_section(self, text: str, metadata: Dict) -> List[MyDocument]:
        """Chunk a section and embed each chunk into a MyDocument."""
        try:
            raw_chunks = self.semantic_splitter.split_text(text)

            valid_chunks = [
                c.strip() for c in (raw_chunks or [])
                if len(c.strip()) >= MIN_CHUNK_SIZE
            ]
            if not valid_chunks:
                return []

            responses: List[EmbeddingResponse] = self.embedding_service.get_embedding_with_uuid(
                data=valid_chunks, chunk_size=32
            )

            if len(responses) != len(valid_chunks):
                logger.error(f"Embedding count mismatch: {len(valid_chunks)} chunks, {len(responses)} responses")
                return []

            documents = []
            for chunk_index, (chunk, r) in enumerate(zip(valid_chunks, responses)):
                chunk_meta = {**metadata, 'chunk_index': chunk_index}
                sparse_vec = SparseVector(indices=r.sparse.indices, values=r.sparse.values) if r.sparse else None
                documents.append(MyDocument(
                    id=r.uuid,
                    text=chunk,
                    embedding=r.embedding,
                    sparse_embedding=sparse_vec,
                    metadata=chunk_meta,
                ))

            return documents

        except Exception as e:
            logger.error(f"Failed to process section: {e}")
            traceback.print_exc()
            return []
