import chromadb
from datetime import datetime
import logging

logging.basicConfig(level=logging.DEBUG)  # or logging.INFO for less verbose logs
logger = logging.getLogger('chromadb')


if __name__ == '__main__':
    # Connecting to db
    client = chromadb.HttpClient(host='localhost', port=8000)

    # creating/getting collection
    collection = client.create_collection(
        name='new_collection', 
        metadata={
            'description': 'Test',
            'created': str(datetime.now())
        },
        get_or_create=True
    )

    # adding data to collection
    # collection.add(
    #     documents=['Tomas Lupic', 'Tomas Mrdka'],
    #     ids=['id1', 'id2']
    # )

    results = collection.query(query_texts=["Tomas"], n_results=2)
    print(f'results: {results}')


