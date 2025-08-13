from etl.Mzdr1DataEtl import Mzdr1DataEtl
from TextEmbeddingService import TextEmbeddingService

if __name__ == "__main__":
    etl = Mzdr1DataEtl("data/MZDR_1_data.csv")
    embedding_service = TextEmbeddingService(ip="localhost", port=8000)
    etl.run(embedding_service)