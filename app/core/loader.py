from pathlib import Path
from typing import Tuple

def load_txt(path: Path) -> Tuple[str, int]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return text, 1


def load_pdf(path: Path) -> Tuple[str, int]:
    try:
        from pypdf import PdfReader
    except ImportError:
        raise RuntimeError("pypdf not installed")

    reader = PdfReader(str(path))
    pages = []

    for page in reader.pages:
        content = page.extract_text() or ""
        pages.append(content)

    full_text = "\n".join(pages)
    return full_text, len(reader.pages)
