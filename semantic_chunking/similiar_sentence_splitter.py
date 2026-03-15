from typing import List

from semantic_chunking.sentence_similarity import SentenceSimilarity
from semantic_chunking.sentence_splitter import SentenceSplitter

class SimilarSentenceSplitter():
    def __init__(self, similarity_model: SentenceSimilarity, sentence_splitter: SentenceSplitter = SentenceSplitter()):
        self.similarity_model = similarity_model
        self.sentence_splitter = sentence_splitter

    def split_text(self, text: str, group_max_sentences: int = 5) -> List[str]:
        sentences = self.sentence_splitter.split(text)

        if len(sentences) == 0:
            return
        
        similarities = self.similarity_model.similarities(sentences)
        groups = [[sentences[0]]]
        for i in range(1, len(sentences)):
            if len(groups[-1]) >= group_max_sentences:
                groups.append([sentences[i]])
            elif similarities[i-1] >= self.similarity_model.similarity_threshold:
                groups[-1].append(sentences[i])
            else:
                groups.append([sentences[i]])

        return [" ".join(g) for g in groups]