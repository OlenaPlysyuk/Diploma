import hashlib
import re
import zlib
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .json_utils import shrink_text

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None


def _pdf_unescape_text(text: str) -> str:
    text = text.replace(r"\(", "(").replace(r"\)", ")").replace(r"\n", "\n").replace(r"\r", "\r")
    text = text.replace(r"\t", "\t").replace(r"\/", "/").replace(r"\\", "\\")
    return text


def _extract_pdf_text_from_bytes(data: bytes) -> str:
    if PdfReader is not None:
        try:
            reader = PdfReader(BytesIO(data))
            pages = []
            for page in reader.pages:
                text = page.extract_text() or ""
                text = text.strip()
                if text:
                    pages.append(text)
            joined = "\n".join(pages).strip()
            if joined:
                return joined
        except Exception:
            pass

    chunks: List[str] = []

    for match in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", data, flags=re.S):
        stream = match.group(1)
        try:
            decoded = zlib.decompress(stream)
        except Exception:
            continue

        try:
            text = decoded.decode("latin-1", errors="ignore")
        except Exception:
            continue

        for txt in re.findall(r"\((.*?)(?<!\\)\)\s*Tj", text, flags=re.S):
            cleaned = _pdf_unescape_text(txt).strip()
            if cleaned:
                chunks.append(cleaned)

        for arr in re.findall(r"\[(.*?)\]\s*TJ", text, flags=re.S):
            parts = re.findall(r"\((.*?)(?<!\\)\)", arr, flags=re.S)
            joined = "".join(_pdf_unescape_text(part) for part in parts).strip()
            if joined:
                chunks.append(joined)

    return "\n".join(chunks)


def load_structure_example(example_path: str) -> Tuple[str, str]:
    if not example_path:
        return "", ""

    path = Path(example_path)
    if not path.exists():
        raise FileNotFoundError(f"Structure example file not found: {path}")

    if path.suffix.lower() == ".pdf":
        content = _extract_pdf_text_from_bytes(path.read_bytes()).strip()
    else:
        content = path.read_text(encoding="utf-8", errors="replace").strip()

    if not content:
        return "", ""

    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return content, digest


def inspect_structure_example(example_path: str) -> Dict[str, Any]:
    content, digest = load_structure_example(example_path)
    preview = shrink_text(content.replace("\r\n", "\n"), 400)
    return {
        "ok": bool(content.strip()),
        "hash": digest,
        "chars": len(content),
        "preview": preview,
    }
