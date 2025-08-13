from enum import Enum
import requests
from typing import Dict, List

from utils.logging_config import get_logger
from utils.utils import Utils

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
            print(f"Request failed: {e}")
            return None

    def embed_text_and_return_result(self, json_data: Dict) -> List[EmbeddingResponse]:
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

            result = self.embed_text_and_return_result(texts)
        else:
            for i, chunk in enumerate(Utils.chunks(data, chunk_size)):
                try:
                    logger.info(f"{i}. chunk processed")

                    temp_arr = self.embed_text_and_return_result({
                        "texts": chunk
                    })

                    result.extend(temp_arr)

                except Exception as e:
                    logger.error(f"Batch failed: {e}")

        return result
    
if __name__ == '__main__':
    obj = TextEmbeddingService('http://127.0.0.1:8000')
    response = obj.get_embedding_with_uuid(['Geralt z Rivie', 'Triss', 'Yennefer', 'Ciri'])
    for i in response:
        print(i.uuid)

    ids = [model.uuid for model in response]
    print(ids)