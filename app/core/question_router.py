import re

DETERMINISTIC_PATTERNS = [
    r"\bhow many words\b",
    r"\bword count\b",
    r"\bnumber of words\b",
    r"\bhow many pages\b",
    r"\bpage count\b",
    r"\bcharacter count\b",
]

def is_deterministic_question(question: str) -> bool:
    q = question.lower()
    return any(re.search(p, q) for p in DETERMINISTIC_PATTERNS)
