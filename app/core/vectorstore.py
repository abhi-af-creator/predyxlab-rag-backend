from typing import List, Dict
from pathlib import Path

import faiss
import numpy as np


class FaissVectorStore:
    """
    Simple FAISS wrapper for storing and searching embeddings.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # cosine similarity (normalized embeddings)
        self.metadata: List[Dict] = []

    def add(self, embeddings: np.ndarray, metadatas: List[Dict]):
        if embeddings.shape[0] != len(metadatas):
            raise ValueError("Embeddings and metadata length mismatch")

        self.index.add(embeddings)
        self.metadata.extend(metadatas)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue

            item = self.metadata[idx].copy()
            item["score"] = float(score)
            results.append(item)

        return results
