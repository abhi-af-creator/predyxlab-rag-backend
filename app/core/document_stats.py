from app.core import state

def total_word_count() -> int:
    return sum(
        len(text.split())
        for text in state.DOCUMENT_TEXTS
    )
