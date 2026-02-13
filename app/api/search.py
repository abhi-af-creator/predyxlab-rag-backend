from fastapi import APIRouter, HTTPException
from app.schemas.search import SearchRequest, SearchResponse
from app.core.embeddings import EmbeddingModel
from app.core import state


router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):

    # ---- Safety checks ----
    if state.VECTOR_STORE is None:
        raise HTTPException(
            status_code=400,
            detail="No document indexed yet"
        )

    if state.ACTIVE_DOC_ID is None:
        raise HTTPException(
            status_code=400,
            detail="No active document selected"
        )

    if not request.query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty"
        )

    # ---- Embed query (local model) ----
    query_embedding = EmbeddingModel.embed_texts([request.query])

    if query_embedding is None or len(query_embedding) == 0:
        raise HTTPException(
            status_code=500,
            detail="Query embedding failed"
        )

    # ---- Retrieve candidates ----
    raw_results = state.VECTOR_STORE.search(
        query_embedding=query_embedding,
        top_k=request.top_k * 3  # retrieve extra then filter
    )

    # ---- Filter by active document ----
    filtered_results = [
        r for r in raw_results
        if r["doc_id"] == state.ACTIVE_DOC_ID
    ][: request.top_k]

    return {"results": filtered_results}
