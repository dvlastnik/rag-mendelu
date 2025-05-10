from enum import Enum
import requests

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
    def from_dict(data: dict):
        if 'uuid' not in data.keys():
            raise ValueError(f"Uuid is missing in provided dictionary: {data}")
        if 'embeddings' not in data.keys():
            raise ValueError(f"Embeddings is missing in provided dictionary: {data}")
        return EmbeddingResponse(uuid=data.get('uuid', str), embedding=data.get('embeddings', list))

class TextEmbeddingService:
    def __init__(self, url: str):
        if ':' not in url:
            raise ValueError(f"Not a valid url address: {url}")
        
        self.url = url

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

    def get_embedding_with_uuid(self, data: list[str] | str) -> list[EmbeddingResponse]:
        texts = {}
        if isinstance(data, list):
            texts = {
                'texts': data
            }
        elif isinstance(data, str):
            texts = {
                'texts': [data]
            }
        response = self._make_request(
            method=HttpMethod.POST,
            endpoint='/embed-text',
            json=texts
        )

        result: list[EmbeddingResponse] = []
        for text in response.get('data'):
            result.append(
                EmbeddingResponse.from_dict(text)
            )

        return result
    
if __name__ == '__main__':
    obj = TextEmbeddingService('http://127.0.0.1:8000')
    response = obj.get_embedding_with_uuid(['Geralt z Rivie', 'Triss', 'Yennefer', 'Ciri'])
    for i in response:
        print(i.uuid)

    ids = [model.uuid for model in response]
    print(ids)