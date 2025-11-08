import os
import traceback
import pathlib
import re
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.config.parser import ConfigParser
from marker.output import text_from_rendered
from typing import List
from pathlib import Path
from docling.document_converter import DocumentConverter

from utils.logging_config import get_logger

logger = get_logger(__name__)

class Utils:
    @staticmethod
    def chunks(array: list, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(array), n):
            yield array[i:i + n]

    @staticmethod
    def _get_safe_name_of_document(path: pathlib.Path) -> str:
        safe_name = path.stem.strip() or "unnamed"
        safe_name = safe_name.replace("/", "_").replace("\\", "_")

        return safe_name

    @staticmethod
    def _get_output_path(path: pathlib.Path, output_folder: str = None, file_type: str = "md") -> pathlib.Path:
        safe_name = Utils._get_safe_name_of_document(path=path)

        if output_folder is None or output_folder == "":
            output_path = path.parent / f"{safe_name}.md"
        else:
            output_folder = os.path.abspath(output_folder)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            
            output_path = pathlib.Path(f"{output_folder}/{safe_name}.{file_type}")

        return output_path
    
    @staticmethod
    def convert_md_table_to_text(table_match: re.Match) -> str:
        """
        Converts a Markdown table (as a regex match object) into a
        linear, readable text string.
        """
        try:
            table_text = table_match.group(0)
            lines = table_text.strip().split('\n')
            
            header: List[str] = []
            data_rows: List[List[str]] = []
            title = ''

            for line in lines:
                if re.match(r'^\s*\|-+\|', line.strip()):
                    continue
                    
                cells = [c.strip().replace('&nbsp;', ' ') for c in line.strip('| \n').split('|')]
                cells = [c for c in cells if c.strip()]
                
                if not cells:
                    continue

                if len(cells) == 1 and not header:
                    title = cells[0]
                elif not header:
                    header = cells
                else:
                    if len(cells) >= len(header):
                        data_rows.append(cells)
                    elif len(cells) < len(header) and len(data_rows) > 0:
                        data_rows[-1][-1] += " " + " ".join(cells)


            if not header or not data_rows:
                return ''

            output_lines: List[str] = []
            if title:
                output_lines.append(f"The table '{title}' shows the following data:")

            row_header_name = header[0]
            col_headers = header[1:]

            for row in data_rows:
                if not row or len(row) < len(header):
                    continue
                    
                row_title = row[0]
                row_parts: List[str] = []
                
                for i, cell_value in enumerate(row[1:]):
                    if i < len(col_headers) and cell_value:
                        row_parts.append(f"{col_headers[i]} is {cell_value.strip()}")
                
                if row_parts:
                    output_lines.append(f"For {row_title}: {'; '.join(row_parts)}.")
                    
            return " ".join(output_lines)
        except Exception as e:
            logger.error(f"Error converting table: {e}\nTable text:\n{table_match.group(0)}")
            return ""


    def clean_md(text: str) -> str:
        """
        Cleans raw Markdown text by removing common academic/PDF artifacts 
        that are unnecessary for RAG.
        """
        if not text:
            return ""

        toc_regex = re.compile(
            r"^#{1,}\s*(\*\*)*(Contents|Table of Contents)(\*\*)*\s*(\n*\|.*)+",
            flags=re.MULTILINE
        )
        cleaned_text = re.sub(toc_regex, '', text)
        
        table_regex = re.compile(
            r'(?:^(?:\|\s*[^|\n]+\s*\|\s*\n))?'
            r'(?:^(?:\|\s*[^|\n]+\s*){2,}\|\s*\n)?'
            r'(^\s*\|(?:[:\-\s|]+)\|*\s*\n)'
            r'((?:^\s*\|.*\|\s*\n?)+)',
            re.MULTILINE
        )
        cleaned_text = re.sub(table_regex, Utils.convert_md_table_to_text, text)
        
        contrib_regex = re.compile(r'#\sContributors.*$', re.DOTALL)
        cleaned_text = re.sub(contrib_regex, '', cleaned_text)
        
        cleaned_text = re.sub(r'</?span[^>]*>', '', cleaned_text)
        
        cleaned_text = re.sub(r'<sup>.*?</sup>', '', cleaned_text)

        cleaned_text = re.sub(r'\[[^\]]*?\]\([^\)]*?\)', '', cleaned_text)
        
        caption_regex = re.compile(r'^\s*\*?\*?\s*(Figure)\s+\d+.*$', re.MULTILINE)
        cleaned_text = re.sub(caption_regex, '', cleaned_text)
        
        source_regex = re.compile(r'^\s*\*?\*?\s*Source:.*$', re.MULTILINE)
        cleaned_text = re.sub(source_regex, '', cleaned_text)
        
        bold_regex = re.compile(r'\*\*([^\*\n]+?)\*\*')
        cleaned_text = re.sub(bold_regex, r'\1', cleaned_text)
        
        italic_regex = re.compile(r'(?<!\*)\*([^\*\n]+?)\*(?!\*)')
        cleaned_text = re.sub(italic_regex, r'\1', cleaned_text)
        
        cleaned_text = re.sub(r'\s?\((Figure|Table)\s\d+\)', '', cleaned_text)
        
        footnote_regex = re.compile(r'^\s*\d+\s+[A-Z].*$', re.MULTILINE)
        cleaned_text = re.sub(footnote_regex, '', cleaned_text)

        url_regex = re.compile(r'(https?|www\.)\S+')
        cleaned_text = re.sub(url_regex, '', cleaned_text)
        
        cleaned_text = re.sub(r'\s+\b(left|right)\)', '', cleaned_text)
        
        cleaned_text = re.sub(r'([,\.])([,\.\s]){2,}', r'\1', cleaned_text)

        stray_num_regex = re.compile(r'^\s*(\d+\s*)+\s*$', re.MULTILINE)
        cleaned_text = re.sub(stray_num_regex, '', cleaned_text)
        
        lines = cleaned_text.split('\n')
        cleaned_lines = [line.strip() for line in lines]
        cleaned_text = '\n'.join(cleaned_lines)
        
        para_fix_regex = re.compile(r'([a-z])\n{2,}([a-z])', re.MULTILINE)
        cleaned_text = re.sub(para_fix_regex, r'\1 \2', cleaned_text)

        cleaned_text = re.sub(r'\n{2,}', '\n\n', cleaned_text) 

        cleaned_text = re.sub(r' +', ' ', cleaned_text)
        
        cleaned_text = re.sub(r'\s+([,\.])', r'\1', cleaned_text)
        
        final_text = cleaned_text.replace('\n', ' ').strip()
        return final_text


    @staticmethod
    def convert_pdf_to_md(path: pathlib.Path, output_folder: str | None = None):
        """Convert one PDF to Markdown."""
        if not path.exists():
            logger.error(f"File not found: {path}")
            return

        if path.suffix.lower() != ".pdf":
            logger.warning(f"Skipping non-PDF file: {path.name}")
            return
        
        output_path = Utils._get_output_path(path=path, output_folder=output_folder, file_type="md")

        if output_path.exists() and output_path.is_file():
            logger.info(f"File already exists! Path: {output_path}")
            return

        try:
            logger.info(f"Processing {path}...")

            converter = DocumentConverter()
            result = converter.convert(str(path))
            text = result.document.export_to_markdown()
            
            # config_dict = {
            #     "disable_image_extraction": True,
            #     "disable_multiprocessing": True,
            #     "output_format": "markdown"
            # }
            
            # config_parser = ConfigParser(config_dict)
            
            # converter = PdfConverter(
            #     artifact_dict=create_model_dict(),
            #     config=config_parser.generate_config_dict()
            # )
            
            # rendered = converter(str(path))
            # text, _, _ = text_from_rendered(rendered)

            output_path.write_text(text, encoding="utf-8")
            logger.info(f"Converted: {output_path}")
            
        except Exception as e:
            logger.error(f"Error converting {path.name}: {e}")
            logger.error(traceback.format_exc())
            return

    def find_files(folder_path: str, file_type: str) -> List[str]:
        """
        Scans a given folder and returns a list of full paths
        to all files ending with file_type.

        Args:
            folder_path (str): Path to the folder to scan.
            file_type (str): File type to search ('pdf', 'csv')

        Returns:
            A list of strings, where each string is the full
            path to a .pdf file.
        """
        search_path = Path(folder_path).resolve()
        
        if not search_path.is_dir():
            logger.error(f"Error: Path '{folder_path}' is not a valid directory.")
            return []

        pdf_files = search_path.glob(f'*.{file_type}')
        
        pdf_file_list = [str(file) for file in pdf_files]
        
        return pdf_file_list
