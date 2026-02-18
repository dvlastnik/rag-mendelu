import spacy
from typing import List

class SentenceSplitter():
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def split(self, text: str) -> List[str]:
        doc = self.nlp(text)

        return [str(sentence).strip() for sentence in doc.sents]