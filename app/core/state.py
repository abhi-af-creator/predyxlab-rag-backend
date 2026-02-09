from typing import Optional
from app.core.vectorstore import FaissVectorStore

VECTOR_STORE: Optional[FaissVectorStore] = None
DOCUMENT_TEXTS: list[str] = []
ACTIVE_DOC_ID = None
