"""
Table extraction and summarization for markdown documents.

Handles extraction of markdown tables, contextual summarization,
and preparation for vector embedding and storage.
"""

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from langchain_ollama import ChatOllama
from utils.logging_config import get_logger

logger = get_logger(__name__)

_TABLE_PATTERN = re.compile(
    r'(?:^|\n)(\|.+\|)\n'
    r'(\|[-:\s|]+\|)\n'
    r'((?:\|.+\|\n?)+)',
    re.MULTILINE
)


@dataclass
class ExtractedTable:
    """Represents an extracted markdown table with context."""
    table_markdown: str
    table_text: str
    summary: str
    preceding_context: str
    row_count: int
    column_count: int
    headers: List[str]
    start_position: int
    end_position: int


class MarkdownTableExtractor:
    """
    Extract and process markdown tables for better RAG context.
    
    Tables are often lost in standard chunking because they:
    1. Have poor semantic meaning when split
    2. Lose row/column relationships
    3. Lack surrounding context
    
    This class extracts tables, summarizes them, and prepares
    them for separate embedding and indexing.
    """
    
    def __init__(self, context_window_chars: int = 1000, use_llm: bool = True, llm_model: str = "granite4:3b"):
        """
        Initialize the table extractor.
        
        Args:
            context_window_chars: Characters to capture before table for context.
            use_llm: Whether to use LLM for table summarization.
            llm_model: LLM model to use for summarization.
        """
        self.context_window_chars = context_window_chars
        self.use_llm = use_llm
        
        if use_llm:
            try:
                self.llm = ChatOllama(
                    model=llm_model,
                    temperature=0,
                    num_ctx=2048,
                    num_predict=2000
                )
                logger.info(f"Initialized LLM for table summarization: {llm_model}")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM: {e}. Falling back to rule-based summarization.")
                self.use_llm = False
                self.llm = None
        else:
            self.llm = None
    
    def extract_tables(self, markdown_text: str) -> List[ExtractedTable]:
        """
        Extract all tables from markdown text.
        
        Args:
            markdown_text: Full markdown document text.
            
        Returns:
            List of ExtractedTable objects with metadata and context.
        """
        tables = []
        
        for match in _TABLE_PATTERN.finditer(markdown_text):
            try:
                table_markdown = match.group(0).strip()
                start_pos = match.start()
                end_pos = match.end()
                
                # Extract headers from first row
                header_row = match.group(1)
                headers = self._parse_table_row(header_row)
                
                # Count rows (exclude separator)
                data_rows_text = match.group(3)
                row_count = len([r for r in data_rows_text.split('\n') if r.strip()])
                column_count = len(headers)
                
                # Get preceding context (section header or nearby text)
                preceding_context = self._extract_preceding_context(
                    markdown_text, start_pos
                )
                
                # Convert table to plain text representation
                table_text = self._table_to_text(table_markdown, headers, preceding_context)
                
                # Generate summary
                summary = self._summarize_table(
                    table_text=table_text,
                    headers=headers,
                    row_count=row_count,
                    preceding_context=preceding_context
                )
                
                extracted = ExtractedTable(
                    table_markdown=table_markdown,
                    table_text=table_text,
                    summary=summary,
                    preceding_context=preceding_context,
                    row_count=row_count,
                    column_count=column_count,
                    headers=headers,
                    start_position=start_pos,
                    end_position=end_pos
                )
                
                tables.append(extracted)
                logger.debug(
                    f"Extracted table: {row_count}x{column_count}, "
                    f"context: '{preceding_context[:50]}...'"
                )
                
            except Exception as e:
                logger.warning(f"Failed to extract table at position {match.start()}: {e}")
                continue
        logger.info(f" - Extracted {len(tables)} tables from markdown")
        return tables
    
    def remove_tables_from_text(self, markdown_text: str, tables: List[ExtractedTable]) -> str:
        """
        Remove tables from markdown text to avoid duplicate processing.
        
        Replaces tables with a placeholder to maintain document flow.
        
        Args:
            markdown_text: Original markdown text.
            tables: List of extracted tables.
            
        Returns:
            Markdown text with tables replaced by placeholders.
        """
        if not tables:
            return markdown_text
        
        # Sort tables by position (reverse order to preserve positions)
        sorted_tables = sorted(tables, key=lambda t: t.start_position, reverse=True)
        
        result = markdown_text
        for table in sorted_tables:
            placeholder = f"\n[TABLE: {table.row_count}x{table.column_count} - {table.headers[0] if table.headers else 'data'}]\n"
            result = (
                result[:table.start_position] + 
                placeholder + 
                result[table.end_position:]
            )
        
        logger.info(f"Removed {len(tables)} tables from text (replaced with placeholders)")
        return result
    
    def _parse_table_row(self, row: str) -> List[str]:
        """Parse a markdown table row into cells."""
        # Remove leading/trailing pipes and split
        cells = row.strip().split('|')
        # Remove empty first/last elements and strip whitespace
        return [cell.strip() for cell in cells if cell.strip()]
    
    def _extract_preceding_context(self, text: str, table_start: int) -> str:
        """
        Extract context before the table (section header or nearby text).
        
        Looks for the nearest markdown header or captures preceding text.
        """
        # Look backwards from table position
        preceding_text = text[max(0, table_start - self.context_window_chars):table_start]
        
        # Try to find the nearest section header
        header_pattern = re.compile(r'^(#{1,4})\s+(.+)$', re.MULTILINE)
        headers = list(header_pattern.finditer(preceding_text))
        
        if headers:
            # Get the last (nearest) header
            last_header = headers[-1]
            header_level = len(last_header.group(1))
            header_text = last_header.group(2).strip()
            
            # Get text between header and table
            text_after_header = preceding_text[last_header.end():].strip()
            
            if text_after_header:
                # Limit to ~200 chars
                context_text = text_after_header[-200:] if len(text_after_header) > 200 else text_after_header
                return f"{header_text} | {context_text}"
            else:
                return header_text
        else:
            # No header found, just use preceding text
            cleaned = preceding_text.strip()
            # Take last sentence or paragraph
            if '.' in cleaned:
                sentences = cleaned.split('.')
                return sentences[-1].strip() if sentences[-1].strip() else sentences[-2].strip()
            return cleaned[-200:] if len(cleaned) > 200 else cleaned
    
    def _table_to_text_llm(self, table_markdown: str, preceding_context: str) -> str:
        """
        Convert markdown table to natural text using LLM.
        
        Args:
            table_markdown: Raw markdown table.
            preceding_context: Context before the table.
            
        Returns:
            Natural language description of the table.
        """
        prompt = f"""Convert this markdown table into clear, readable text. 

        Context: {preceding_context if preceding_context else 'Data table'}

        Table:
        {table_markdown}

        Provide a concise summary followed by the complete table data in a readable format. Keep all data but make it easier to read as continuous text. Format each row on a new line."""
        
        try:
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            logger.warning(f"LLM table conversion failed: {e}. Using rule-based fallback.")
            return self._table_to_text_rule_based(table_markdown, self._parse_table_row(table_markdown.split('\n')[0]))
    
    def _table_to_text_rule_based(self, table_markdown: str, headers: List[str]) -> str:
        """
        Convert markdown table to plain text representation (rule-based fallback).
        
        Format: Header1: value | Header2: value ...
        """
        lines = table_markdown.strip().split('\n')
        
        # Skip header and separator rows
        data_rows = []
        for i, line in enumerate(lines):
            # Skip first two rows (header and separator)
            if i < 2:
                continue
            # Skip separator-like rows
            if re.match(r'^\s*\|[\s\-:|]+\|\s*$', line):
                continue
            data_rows.append(line)
        
        # Parse each data row
        text_parts = []
        for row in data_rows:
            cells = self._parse_table_row(row)
            if not cells:
                continue
            
            # Create "Header: Value" pairs
            row_parts = []
            for i, cell in enumerate(cells):
                header = headers[i] if i < len(headers) else f"Column{i+1}"
                if cell:  # Skip empty cells
                    row_parts.append(f"{header}: {cell}")
            
            if row_parts:
                text_parts.append(" | ".join(row_parts))
        
        return "\n".join(text_parts)
    
    def _table_to_text(self, table_markdown: str, headers: List[str], preceding_context: str = "") -> str:
        """
        Convert markdown table to text using LLM or rule-based approach.
        
        Args:
            table_markdown: Raw markdown table.
            headers: Parsed headers.
            preceding_context: Context before the table.
            
        Returns:
            Text representation of the table.
        """
        if self.use_llm and self.llm:
            return self._table_to_text_llm(table_markdown, preceding_context)
        else:
            return self._table_to_text_rule_based(table_markdown, headers)
    
    def _summarize_table(
        self,
        table_text: str,
        headers: List[str],
        row_count: int,
        preceding_context: str
    ) -> str:
        """
        Generate a concise summary of the table for embedding.
        
        If LLM was used, table_text already contains natural language.
        Otherwise, add context and structure.
        
        Args:
            table_text: Plain text representation of table.
            headers: List of column headers.
            row_count: Number of data rows.
            preceding_context: Context text before the table.
            
        Returns:
            Formatted summary text suitable for embedding.
        """
        # If LLM was used, the text is already well-formatted
        if self.use_llm and self.llm:
            return table_text
        
        # Rule-based formatting
        summary_parts = []
        
        if preceding_context:
            summary_parts.append(f"Context: {preceding_context}")
        
        summary_parts.append(f"Columns: {', '.join(headers)}")
        summary_parts.append(table_text)
        
        return "\n".join(summary_parts)
    
    def _chunk_table_by_rows(
        self,
        table_text: str,
        headers: List[str],
        preceding_context: str,
        rows_per_chunk: int = 15,
        max_chars: int = 2000
    ) -> List[str]:
        """
        Split a large table into smaller chunks by rows.
        
        Each chunk maintains:
        - Context
        - Column headers
        - A subset of rows
        
        Args:
            table_text: Plain text representation of table.
            headers: List of column headers.
            preceding_context: Context text before the table.
            rows_per_chunk: Maximum rows per chunk.
            max_chars: Maximum characters per chunk.
            
        Returns:
            List of table chunk summaries.
        """
        text_lines = table_text.strip().split('\n')
        
        # If table is small enough, return as single chunk
        if len(text_lines) <= rows_per_chunk:
            summary = self._summarize_table(table_text, headers, len(text_lines), preceding_context)
            if len(summary) <= max_chars:
                return [summary]
        
        # Split into chunks
        chunks = []
        for i in range(0, len(text_lines), rows_per_chunk): 
            chunk_rows = text_lines[i:i + rows_per_chunk]
            chunk_text = '\n'.join(chunk_rows)
            
            # Build chunk summary
            chunk_parts = []
            
            # For LLM-generated text, keep it simple
            if self.use_llm and self.llm:
                if preceding_context and i == 0:  # Only add context to first chunk
                    chunk_parts.append(f"Context: {preceding_context}")
                if len(text_lines) > rows_per_chunk:
                    chunk_parts.append(f"Part {i//rows_per_chunk + 1} (rows {i+1}-{i+len(chunk_rows)} of {len(text_lines)})")
                chunk_parts.append(chunk_text)
            else:
                # Rule-based formatting
                if preceding_context:
                    chunk_parts.append(f"Context: {preceding_context}")
                chunk_parts.append(f"Columns: {', '.join(headers)}")
                chunk_parts.append(f"Rows {i+1}-{i+len(chunk_rows)} of {len(text_lines)}")
                chunk_parts.append(chunk_text)
            
            chunk_summary = '\n'.join(chunk_parts)
            
            # If still too long, truncate
            if len(chunk_summary) > max_chars:
                chunk_summary = chunk_summary[:max_chars] + "\n... [truncated]"
            
            chunks.append(chunk_summary)
        
        return chunks


class TableProcessor:
    """
    High-level processor for integrating table extraction into ETL pipeline.
    
    Handles extraction, summarization, and preparation of tables
    for embedding and storage alongside regular text chunks.
    """
    
    def __init__(self, extractor: Optional[MarkdownTableExtractor] = None):
        """
        Initialize the table processor.
        
        Args:
            extractor: Optional custom extractor instance.
        """
        self.extractor = extractor or MarkdownTableExtractor()
    
    def process_document(
        self,
        markdown_text: str,
        base_metadata: Dict
    ) -> Tuple[str, List[Dict]]:
        """
        Process a markdown document: extract tables and prepare table documents.
        
        Large tables are automatically chunked by rows to prevent embedding service overload.
        
        Args:
            markdown_text: Full markdown text.
            base_metadata: Base metadata to attach to table documents.
            
        Returns:
            Tuple of (text_without_tables, list_of_table_documents)
            where each table document is a dict with:
                - text: Summary text for embedding
                - metadata: Enhanced metadata including table info
        """
        tables = self.extractor.extract_tables(markdown_text)
        
        if not tables:
            return markdown_text, []
        
        logger.info(f'Found {len(tables)} tables!')
        # Remove tables from original text
        text_without_tables = self.extractor.remove_tables_from_text(
            markdown_text, tables
        )
        
        # Create table documents (with chunking for large tables)
        table_documents = []
        for i, table in enumerate(tables):
            # Chunk table if needed
            table_chunks = self.extractor._chunk_table_by_rows(
                table_text=table.table_text,
                headers=table.headers,
                preceding_context=table.preceding_context,
                rows_per_chunk=15,
                max_chars=2000
            )
            
            # Create a document for each chunk
            for chunk_idx, chunk_text in enumerate(table_chunks):
                table_metadata = base_metadata.copy()
                table_metadata.update({
                    'is_table': True,
                    'table_index': i,
                    'table_chunk': chunk_idx if len(table_chunks) > 1 else None,
                    'table_total_chunks': len(table_chunks) if len(table_chunks) > 1 else None,
                    'table_rows': table.row_count,
                    'table_columns': table.column_count,
                    'table_headers': table.headers,
                    'table_context': table.preceding_context,
                })
                
                table_documents.append({
                    'text': chunk_text,
                    'metadata': table_metadata,
                    'original_markdown': table.table_markdown,  # For debugging
                })
        
        logger.info(
            f"Processed document: extracted {len(tables)} tables into {len(table_documents)} chunks, "
            f"text length reduced from {len(markdown_text)} to {len(text_without_tables)}"
        )
        
        return text_without_tables, table_documents
