from fastapi import APIRouter, HTTPException
from app.schemas.chat import ChatRequest, ChatResponse
from app.core.embeddings import EmbeddingModel
from app.core import state
from app.llm.ollama_client import OllamaClient

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    if state.VECTOR_STORE is None:
        raise HTTPException(status_code=400, detail="Vector store not initialized")

    if state.ACTIVE_DOC_ID is None:
        raise HTTPException(status_code=400, detail="No document indexed yet")

    # Embed query (FIXED)
    query_embedding = EmbeddingModel.embed_texts(
        [request.question]
    )[0]

    # Retrieve
    raw_results = state.VECTOR_STORE.search(
        query_embedding=query_embedding,
        top_k=request.top_k * 3
    )

    # Filter by active document
    results = [
        r for r in raw_results
        if r["doc_id"] == state.ACTIVE_DOC_ID
    ][: request.top_k]

    if not results:
        return {
            "answer": "I don't know based on the document.",
            "sources": []
        }

    context = "\n\n".join(r["text"] for r in results)

    prompt = f"""
You are a precise research assistant.

Rules:
- Use ONLY the information provided in the context.
- Do NOT use prior knowledge.
- Do NOT guess.
- If the answer is not explicitly stated, reply: "I don't know based on the document."

Context:
{context}

Question:
{request.question}

Answer (concise, factual):
""".strip()

    answer = OllamaClient.generate(prompt)

    return {
        "answer": answer.strip(),
        "sources": [
            f'{r["doc_id"]}:{r["chunk_id"]}'
            for r in results
        ]
    }
