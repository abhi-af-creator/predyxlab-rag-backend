from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.health import router as health_router
from app.api.ingest import router as ingest_router
from app.api.search import router as search_router
from app.api.chat import router as chat_router
from app.api import search

from app.core.embeddings import EmbeddingModel




app = FastAPI(
    title="PredyxLab Research Assistant API",
    version="0.1.0"
)

@app.on_event("startup")
def load_models():
    EmbeddingModel.get_model()

# -------------------- CORS --------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://orange-moss-08315ef00.2.azurestaticapps.net",
                  "https://black-grass-09ef5ab00.2.azurestaticapps.net",
                  "https://calm-meadow-06bb15200.2.azurestaticapps.net"],   # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- ROUTERS --------------------
app.include_router(health_router)
app.include_router(ingest_router)
app.include_router(search_router)
app.include_router(chat_router)

# -------------------- ROOT --------------------
@app.get("/")
def root():
    return {
        "service": "predyxlab-rag",
        "status": "running"
    }

@app.get("/health")
def health():
    return{
        "service":"health",
        "status":"ok"
    }

@app.get("/test-ollama")
def test_ollama():
    from app.llm.ollama_client import OllamaClient
    return{"response": OllamaClient.generate("How are you?")}
