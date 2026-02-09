from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from app.config import EMBEDDING_MODEL_NAME


class EmbeddingModel:
    """
    Singleton-style embedding loader.
    Model loads once per process.
    """

    _model = None

    @classmethod
    def get_model(cls) -> SentenceTransformer:
        if cls._model is None:
            cls._model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        return cls._model

    @classmethod
    def embed_texts(cls, texts: List[str]) -> np.ndarray:
        model = cls.get_model()
        embeddings = model.encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        return embeddings
