import uuid
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.config import UPLOAD_DIR
from app.core.loader import load_pdf, load_txt
from app.core.chunker import chunk_text
from app.core.embeddings import EmbeddingModel
from app.core.vectorstore import FaissVectorStore
from app.core import state


router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/ingest")
async def ingest_document(file: UploadFile = File(..., description="PDF or TXT file, max 10MB")):
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

    contents = await file.read()

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="File too large. Maximum allowed size is 10MB."
        )

    # IMPORTANT: reset file pointer after read
    await file.seek(0)



    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    suffix = Path(file.filename).suffix.lower()

    if suffix not in {".pdf", ".txt"}:
        raise HTTPException(
            status_code=400,
            detail="Only PDF and TXT files are supported"
        )

    doc_id = str(uuid.uuid4())
    state.ACTIVE_DOC_ID = doc_id
    dest = UPLOAD_DIR / f"{doc_id}{suffix}"

    with dest.open("wb") as f:
        f.write(await file.read())

    if suffix == ".pdf":
        text, pages = load_pdf(dest)
    else:
        text, pages = load_txt(dest)

    # ✅ Store full document text ONCE (for deterministic queries)
    state.DOCUMENT_TEXTS.append(text)

    # Chunking
    chunks = chunk_text(text)
    chunk_texts = [c["text"] for c in chunks]

    # Embeddings
    embeddings = EmbeddingModel.embed_texts(chunk_texts)

    # Vector store init
    if state.VECTOR_STORE is None:
        state.VECTOR_STORE = FaissVectorStore(dim=embeddings.shape[1])

    # Metadata
    metadatas = [
        {
            "doc_id": doc_id,
            "chunk_id": c["chunk_id"],
            "text": c["text"],
            "start": c["start"],
            "end": c["end"]
        }
        for c in chunks
    ]

    # Add to vector store
    state.VECTOR_STORE.add(embeddings, metadatas)

    return {
        "doc_id": doc_id,
        "filename": file.filename,
        "pages": pages,
        "chars": len(text),
        "chunks": len(chunks),
        "embedding_dim": embeddings.shape[1],
        "indexed": True
    }
