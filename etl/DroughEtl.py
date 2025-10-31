import json
import traceback
from typing import List, Tuple
import pandas as pd
import pathlib
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import ast

from etl.BaseEtl import BaseEtl, ETLState
from database.base.MyDocument import MyDocument
from utils.Utils import Utils
from utils.logging_config import get_logger

logger = get_logger(__name__)

class DroughEtl(BaseEtl):
    SECTION_RE = re.compile(
        r'^\s*(?P<hashes>#{1,})\s*(?P<title>.+?)\s*$'   # heading line (allow leading spaces)
        r'(?:\r?\n)+'                                   # one or more newlines after heading line
        r'(?P<body>.*?)(?=^\s*#{1,}\s+|\Z)',            # non-greedy body until next heading or EOF
        flags=re.DOTALL | re.MULTILINE
    )

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

    def _extract_sections(self, markdown_text: str) -> List[Tuple[str, str, str, str]]:
        html_regex = r"<.*?>"

        FORBIDDEN_TITLE_KEYWORDS = {
            "cite", "citing", "credits", "endnotes", "members", "references", "content"
        }

        sections = []
        for m in self.SECTION_RE.finditer(markdown_text):
            level = m.group('hashes')
            title = m.group('title').strip()
            body = m.group('body').rstrip()

            if len(body) <= 20:
                continue

            title = re.sub(html_regex, "", title).strip()
            title = title.replace("**", "")
            title_lower = title.lower()

            if any(word in title_lower for word in FORBIDDEN_TITLE_KEYWORDS):
                continue

            body = re.sub(html_regex, "", body).strip()

            updated_body = self._merge_body(body)

            metadata = {
                "source": self.file.stem,
                "title": title
            }

            if body != "" and "cite" not in title and "citing" not in title and "credits" not in title and "endnotes" not in title and "members" not in title and len(body) > 20:
                sections.append((level, title, updated_body, metadata))
        return sections

    def _transform_with_model(self) -> None:
        """
        Experimental function, uses pretrained model to chunk text.
        """
        path_to_md = self.get_file_path(only_folder=False, chunk_index=None)
        md_text = path_to_md.read_text(encoding="utf-8")
        
        sections = self._extract_sections(md_text)

        # COPY PASTE
        model_name = "chentong00/propositionizer-wiki-flan-t5-large"
        device = "mps" if torch.mps.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

        for index, (level, heading, body, metadata) in enumerate(sections[0:3]):
            input_text = body
            
            input_ids = tokenizer(input_text, max_length=2048, truncation=True, return_tensors="pt").input_ids
            outputs = model.generate(input_ids.to(device), max_new_tokens=2048).cpu()

            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Type: {type(output_text)}")
            json_match = re.search(r'\[\s*".*?\]\s*$', output_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                try:
                    chunked_sentences = json.loads(json_str)
                except json.JSONDecodeError:
                    # Fallback: split manually
                    logger.error(f"Failed to parse: {output_text[:200]}")
                    chunked_sentences = []

                if len(chunked_sentences) > 0:
                    with open(self.get_file_path(), "w", encoding="utf-8") as f:
                        f.write(f"{level} {heading}")
                        f.write("\n")
                        f.write("\n".join(chunked_sentences))
    
    def transform(self) -> None:
        try:
            # Testing
            # path_to_folder = self.get_file_path(only_folder=True)
            # number_of_md_files = len(Utils.find_files(path_to_folder, "md"))
            # if number_of_md_files >= 20:
            #     logger.info(f"File: {self.file} was already transformed. Number of files found: {number_of_md_files}")
            #     return

            path_to_md = self.get_file_path(only_folder=False)
            md_text = path_to_md.read_text(encoding="utf-8")
            
            sections = self._extract_sections(md_text)
            for index, (level, heading, body, metadata) in enumerate(sections):
                    clean_body = Utils.clean_md(body)
                    # Real
                    if clean_body == '':
                        continue
                    
                    chunk_and_embed_result = self.embedding_service.chunk_and_embed(data=clean_body)
                    for chunk in chunk_and_embed_result:
                        self.documents.append(MyDocument(id=chunk.embed_text.uuid, text=chunk.text, embedding=chunk.embed_text.embedding, metadata=metadata))
                    
                    # Testing
                    # result_of_chunking = self.embedding_service.chunk_text(clean_body)
                    # with open(self.get_file_path(only_folder=False, chunk_index=index), 'w', encoding='utf-8') as f:
                    #     f.write(f"{level} {heading}\n")
                    #     f.write("\n")
                    #     for chunk in result_of_chunking.sentences:
                    #         if ("Figure" in chunk and len(chunk) < 20) or chunk[0] == '[' or len(chunk) < 10 or chunk[0] == '-':
                    #             continue
                    #         f.write(f"{chunk}\n")
                    #         f.write("\n")
                    #     f.write(f"Metadata: {str(metadata)}")
            
            logger.info(f"File: {self.file} was transformed!")
            logger.info(f"Number of documents: {len(self.documents)}")
            self.state = ETLState.TRANSFORMED
            return
        except Exception as e:
            logger.exception("Error during transform step")
            traceback.print_exc()
            self.state = ETLState.FAILED
            return


        


        

        
    
    