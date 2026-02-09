from typing import List, Dict


def chunk_text(
    text: str,
    chunk_size: int = 800,
    overlap: int = 120
) -> List[Dict]:
    """
    Split text into overlapping chunks.

    Returns:
        [
          {
            "chunk_id": int,
            "text": str,
            "start": int,
            "end": int
          }
        ]
    """

    if not text or not text.strip():
        return []

    chunks = []
    start = 0
    chunk_id = 0
    length = len(text)

    while start < length:
        end = start + chunk_size
        chunk_text = text[start:end]

        chunks.append({
            "chunk_id": chunk_id,
            "text": chunk_text,
            "start": start,
            "end": min(end, length)
        })

        chunk_id += 1
        start = end - overlap

        if start < 0:
            start = 0

    return chunks
