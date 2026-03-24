"""
Table extraction for markdown documents.

Extracts markdown tables as individual row documents with column values
stored as typed metadata, ready for vector embedding and filtering.
"""

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
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
    """Represents an extracted markdown table."""
    table_markdown: str
    headers: List[str]
    rows: List[List[str]]
    start_position: int
    end_position: int


class MarkdownTableExtractor:
    """
    Extracts markdown tables and converts them to per-row documents.

    Each data row becomes a separate document where column values
    are stored as typed metadata (int / float / lowercase str).
    """

    def extract_tables(self, markdown_text: str) -> List[ExtractedTable]:
        """
        Extract all tables from markdown text.

        Args:
            markdown_text: Full markdown document text.

        Returns:
            List of ExtractedTable objects.
        """
        tables = []

        for match in _TABLE_PATTERN.finditer(markdown_text):
            try:
                headers = self._parse_row(match.group(1))

                rows = []
                for line in match.group(3).strip().split('\n'):
                    if not line.strip():
                        continue
                    cells = self._parse_row(line)
                    if cells:
                        rows.append(cells)

                tables.append(ExtractedTable(
                    table_markdown=match.group(0).strip(),
                    headers=headers,
                    rows=rows,
                    start_position=match.start(),
                    end_position=match.end(),
                ))
                logger.debug(f"Extracted table: {len(rows)} rows × {len(headers)} columns")

            except Exception as e:
                logger.warning(f"Failed to extract table at position {match.start()}: {e}")

        logger.info(f"Extracted {len(tables)} tables from markdown")
        return tables

    def remove_tables_from_text(self, markdown_text: str, tables: List[ExtractedTable]) -> str:
        """
        Remove extracted tables from markdown, replacing each with a short placeholder.

        Args:
            markdown_text: Original markdown text.
            tables: Tables previously returned by extract_tables().

        Returns:
            Markdown text with tables replaced by placeholders.
        """
        if not tables:
            return markdown_text

        result = markdown_text
        for table in sorted(tables, key=lambda t: t.start_position, reverse=True):
            placeholder = (
                f"\n[TABLE: {len(table.rows)} rows × {len(table.headers)} columns — "
                f"headers: {', '.join(table.headers)}]\n"
            )
            result = result[:table.start_position] + placeholder + result[table.end_position:]

        return result

    def _parse_row(self, row: str) -> List[str]:
        """Parse a markdown table row into cell values, stripping whitespace."""
        cells = row.strip().split('|')
        return [cell.strip() for cell in cells if cell.strip()]

    def _infer_type(self, value: str):
        """
        Try to convert a string value to int or float.
        Falls back to lowercase string.

        Examples:
            "25"   → 25
            "9.4"  → 9.4
            "Male" → "male"
        """
        stripped = value.strip()
        try:
            return int(stripped)
        except ValueError:
            pass
        try:
            return float(stripped)
        except ValueError:
            pass
        return stripped.lower()


class TableProcessor:
    """
    High-level processor that integrates table extraction into the ETL pipeline.

    Produces one document per table row with column values as typed metadata.

    Example
    -------
    Markdown table::

        | Name    | Age | Gender |
        |---------|-----|--------|
        | David   | 25  | Male   |
        | Natalie | 24  | Female |

    Produces two documents::

        {"text": "| David | 25 | Male |",    "metadata": {"name": "david",   "age": 25, "gender": "male",   "is_table": True}}
        {"text": "| Natalie | 24 | Female |", "metadata": {"name": "natalie", "age": 24, "gender": "female", "is_table": True}}
    """

    def __init__(self, extractor: Optional[MarkdownTableExtractor] = None):
        self.extractor = extractor or MarkdownTableExtractor()

    def process_document(
        self,
        markdown_text: str,
        base_metadata: Dict,
    ) -> Tuple[str, List[Dict]]:
        """
        Extract tables from markdown and convert each data row to a document.

        Args:
            markdown_text: Full markdown text.
            base_metadata: Metadata merged into every row document
                           (e.g. ``{"source": "my_file", "file_type": "md"}``).

        Returns:
            A tuple ``(text_without_tables, row_documents)`` where:

            * ``text_without_tables`` – original markdown with tables replaced
              by short placeholders.
            * ``row_documents`` – list of dicts, one per data row::

                {
                    "text":     "| David | 25 | Male |",
                    "metadata": {
                        "is_table": True,
                        "name":     "david",   # header key → typed cell value
                        "age":      25,
                        "gender":   "male",
                        ...base_metadata fields...
                    }
                }
        """
        tables = self.extractor.extract_tables(markdown_text)

        if not tables:
            return markdown_text, []

        text_without_tables = self.extractor.remove_tables_from_text(markdown_text, tables)

        row_documents = []
        for table in tables:
            for row_cells in table.rows:
                row_col_meta: Dict = {}
                for i, header in enumerate(table.headers):
                    key = header.lower().replace(' ', '_')
                    raw = row_cells[i] if i < len(row_cells) else ''
                    row_col_meta[key] = self.extractor._infer_type(raw)
            
                row_meta = {**row_col_meta, **base_metadata, 'is_table': True}

                row_documents.append({
                    'text': '| ' + ' | '.join(row_cells) + ' |',
                    'metadata': row_meta,
                })

        logger.info(
            f"Tables → {len(row_documents)} row documents  "
            f"(markdown: {len(markdown_text):,} → {len(text_without_tables):,} chars after removal)"
        )

        return text_without_tables, row_documents
