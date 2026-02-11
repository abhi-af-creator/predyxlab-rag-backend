import requests

ML_SERVICE_URL = "http://20.168.119.134:8001/embed"

class EmbeddingModel:

    @staticmethod
    def embed_texts(texts):
        response = requests.post(
            ML_SERVICE_URL,
            json={"texts": texts},
            timeout=120
        )
        response.raise_for_status()
        return response.json()["embeddings"]
