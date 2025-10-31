from enum import Enum
import requests
from typing import Dict, List

from utils.logging_config import get_logger
from utils.Utils import Utils

logger = get_logger(__name__)

class HttpMethod(Enum):
    GET = "GET"
    POST = "POST"
    DELETE = "DELETE"
    PUT = "PUT"

class EmbeddingResponse:
    def __init__(self, uuid, embedding):
        self.uuid = uuid
        self.embedding = embedding

    def __str__(self):
        return f"Uuid: {self.uuid}, Embedding: {self.embedding}"

    @staticmethod
    def from_dict(data: Dict) -> 'EmbeddingResponse':
        if 'uuid' not in data.keys():
            raise ValueError(f"Uuid is missing in provided dictionary: {data}")
        if 'embeddings' not in data.keys():
            raise ValueError(f"Embeddings is missing in provided dictionary: {data}")
        return EmbeddingResponse(uuid=data.get('uuid', str), embedding=data.get('embeddings', list))
    
    @staticmethod
    def from_list_of_dicts(data: List[Dict]) -> List['EmbeddingResponse']:
        result: List['EmbeddingResponse'] = []

        for d in data['data']:
            result.append(EmbeddingResponse.from_dict(d))

        return result
    
class ChunkTextResponse:
    def __init__(self, sentences):
        self.sentences = sentences

    def __str__(self):
        return str(self.sentences)

    @staticmethod
    def from_dict(data: Dict) -> 'ChunkTextResponse':
        if 'sentences' not in data.keys():
            raise ValueError(f"sentences missing in provided dictionary: {data}")
        return ChunkTextResponse(sentences=data.get('sentences', list))
    
class ChunkAndEmbedResponse:
    def __init__(self, text, embed_text: EmbeddingResponse):
        self.text = text
        self.embed_text = embed_text
    
    def __str__(self):
        return f"Text: {self.text}, Embed Text dict: {self.embed_text}"
    
    @staticmethod
    def from_dict(data: Dict) -> 'ChunkAndEmbedResponse':
        if any(key not in data.keys() for key in ['text', 'embed_text']):
            raise ValueError(f"Text or embed_text is missing in dictionary: {data}")
        return ChunkAndEmbedResponse(text=data.get('text', str), embed_text=EmbeddingResponse.from_dict(data.get('embed_text', dict)))
    
    @staticmethod
    def from_dict_list(data: Dict) -> List['ChunkAndEmbedResponse']:
        result: List['ChunkAndEmbedResponse'] = []

        for d in data['data']:
            result.append(ChunkAndEmbedResponse.from_dict(d))

        return result

class TextEmbeddingService:
    def __init__(self, ip: str, port: int):
        self.url = "http://" + ip + ":" + str(port)

    def _make_request(self, method: HttpMethod, endpoint: str, params=None, data=None, headers=None, json=None, timeout=10):
        try:
            response = requests.request(
                method=method.value,
                url=self.url + endpoint,
                params=params,
                data=data,
                headers=headers,
                json=json,
                timeout=timeout
            )
            response.raise_for_status()
            return response.json() if 'application/json' in response.headers.get('Content-Type', '') else response.text
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}, data: {json}")
            raise requests.exceptions.RequestException()

    def _embed_text_and_return_result(self, json_data: Dict) -> List[EmbeddingResponse]:
        response = self._make_request(
            method=HttpMethod.POST,
            endpoint='/embed-text',
            json=json_data
        )

        return EmbeddingResponse.from_list_of_dicts(response)

    def get_embedding_with_uuid(self, data: List[str] | str, chunk_size=None) -> List[EmbeddingResponse]:
        result: List[EmbeddingResponse] = []

        if chunk_size is None or chunk_size == 0:
            texts = {}

            if isinstance(data, list):
                texts = {
                    'texts': data
                }
            elif isinstance(data, str):
                texts = {
                    'texts': [data]
                }

            result = self._embed_text_and_return_result(texts)
        else:
            for i, chunk in enumerate(Utils.chunks(data, chunk_size)):
                try:
                    logger.info(f"{i}. chunk processed")

                    temp_arr = self._embed_text_and_return_result({
                        "texts": chunk
                    })

                    result.extend(temp_arr)

                except Exception as e:
                    logger.error(f"Batch failed: {e}")

        return result
    
    def chunk_text(self, data: str) -> ChunkTextResponse:
        json = {
            "text": data
        }

        response = self._make_request(
            method=HttpMethod.POST,
            endpoint="/chunk-by-similarity",
            json=json
        )

        return ChunkTextResponse.from_dict(response)
    
    def chunk_and_embed(self, data: str) -> List[ChunkAndEmbedResponse]:
        json = {
            "text": data
        }

        response = self._make_request(
            method=HttpMethod.POST,
            endpoint="/chunk-and-embed",
            json=json
        )

        return ChunkAndEmbedResponse.from_dict_list(response)