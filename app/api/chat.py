from fastapi import APIRouter, HTTPException
from app.schemas.chat import ChatRequest, ChatResponse
from app.core.embeddings import EmbeddingModel
from app.core import state
from app.llm.ollama_client import OllamaClient

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if state.VECTOR_STORE is None or state.ACTIVE_DOC_ID is None:
        raise HTTPException(status_code=400, detail="No document indexed yet")

    # Embed query
    query_embedding = EmbeddingModel.embed_texts([request.question])

    # Retrieve
    raw_results = state.VECTOR_STORE.search(
        query_embedding=query_embedding,
        top_k=request.top_k * 3  # overfetch
    )

    # 🔥 FILTER BY ACTIVE DOCUMENT
    results = [
        r for r in raw_results
        if r["doc_id"] == state.ACTIVE_DOC_ID
    ][: request.top_k]

    if not results:
        return {
            "answer": "No relevant information found in the current document.",
            "sources": []
        }

    # Build context
    context = "\n\n".join(r["text"] for r in results)

    prompt = f"""
You are a research assistant.
Answer ONLY using the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{request.question}

Answer:
""".strip()

    answer = OllamaClient.generate(prompt)

    return {
        "answer": answer.strip(),
        "sources": [
            f'{r["doc_id"]}:{r["chunk_id"]}'
            for r in results
        ]
    }
