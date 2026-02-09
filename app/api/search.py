from fastapi import APIRouter, HTTPException
from app.schemas.search import SearchRequest, SearchResponse
from app.core.embeddings import EmbeddingModel
from app.core import state

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    if state.VECTOR_STORE is None or state.ACTIVE_DOC_ID is None:
        raise HTTPException(status_code=400, detail="No document indexed yet")

    query_embedding = EmbeddingModel.embed_texts([request.query])

    raw_results = state.VECTOR_STORE.search(
        query_embedding=query_embedding,
        top_k=request.top_k * 3
    )

    results = [
        r for r in raw_results
        if r["doc_id"] == state.ACTIVE_DOC_ID
    ][: request.top_k]

    return {"results": results}

