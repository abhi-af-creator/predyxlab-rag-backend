import requests


class OllamaClient:
    BASE_URL = "http://20.168.119.134:11434"
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
