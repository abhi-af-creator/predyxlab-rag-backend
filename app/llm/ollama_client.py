import requests
import os


class OllamaClient:
    BASE_URL = os.getenv("OLLAMA_BASE_URL","http://localhost:11434")
    MODEL = "llama3"

    @classmethod
    def generate(cls, prompt: str) -> str:
        response = requests.post(
            f"{cls.BASE_URL}/api/generate",
            json={
                "model": cls.MODEL,
                "prompt": prompt,
                "stream": False
            },
            timeout=300
        )

        response.raise_for_status()
        data = response.json()
        return data["response"].strip()
