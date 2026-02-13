import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    _model = None

    @classmethod
    def get_model(cls):
        if cls._model is None:
            cls._model = SentenceTransformer("all-MiniLM-L6-v2")
        return cls._model

    @classmethod
    def embed_texts(cls, texts):
        model = cls.get_model()
        embeddings = model.encode(
            texts,
            normalize_embeddings=True
        )
        return np.array(embeddings).astype("float32")
