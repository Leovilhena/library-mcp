"""PDF/EPUB text extraction and chunking.

No network, no execution of anything from the document -- this only ever
reads text out of a file. The one real risk class for a parser like this is
a malformed/adversarial file crashing or hanging the process, which is why
this runs in its own minimal, resource-limited container rather than inside
anything that also handles execution or network access.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


class ParseError(Exception):
    """A document could not be parsed."""


@dataclass(frozen=True, slots=True)
class TextBlock:
    section: str | None
    text: str


_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t]+")
_BLANKLINES_RE = re.compile(r"\n{3,}")


def _strip_html(html: str) -> str:
    text = _TAG_RE.sub(" ", html)
    text = _WS_RE.sub(" ", text)
    text = _BLANKLINES_RE.sub("\n\n", text)
    return text.strip()


def extract_pdf(path: Path) -> list[TextBlock]:
    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        msg = f"could not open PDF {path.name}: {exc}"
        raise ParseError(msg) from exc
    blocks: list[TextBlock] = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # a single bad page shouldn't sink the whole book
            text = ""
            blocks.append(TextBlock(section=f"page {i + 1}", text=f"[unreadable page: {exc}]"))
            continue
        text = text.strip()
        if text:
            blocks.append(TextBlock(section=f"page {i + 1}", text=text))
    if not blocks:
        msg = f"{path.name}: no extractable text (likely a scanned/image-only PDF -- OCR not supported)"
        raise ParseError(msg)
    return blocks


def extract_epub(path: Path) -> list[TextBlock]:
    try:
        import ebooklib
        from ebooklib import epub
    except ImportError as exc:  # pragma: no cover - import-time dependency check
        msg = "ebooklib is not installed"
        raise ParseError(msg) from exc
    try:
        book = epub.read_epub(str(path))
    except Exception as exc:
        msg = f"could not open EPUB {path.name}: {exc}"
        raise ParseError(msg) from exc
    blocks: list[TextBlock] = []
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        if isinstance(item, epub.EpubNav):
            # The nav/TOC document is a list of chapter titles, not content --
            # measured to actively hurt retrieval: its superficial keyword
            # overlap with a chapter title outranked a genuinely relevant
            # chunk from a *different* book in a real test (see
            # docs/architecture/library-mcp.md).
            continue
        raw = item.get_content().decode("utf-8", errors="replace")
        text = _strip_html(raw)
        if text:
            blocks.append(TextBlock(section=item.get_name(), text=text))
    if not blocks:
        msg = f"{path.name}: no extractable text found"
        raise ParseError(msg)
    return blocks


def extract(path: Path) -> list[TextBlock]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf(path)
    if suffix == ".epub":
        return extract_epub(path)
    msg = f"unsupported file type: {suffix}"
    raise ParseError(msg)


@dataclass(frozen=True, slots=True)
class Chunk:
    section: str | None
    text: str


def chunk_blocks(blocks: list[TextBlock], chunk_chars: int, overlap_chars: int) -> list[Chunk]:
    """Split each block into overlapping chunks, keeping the block's section label.

    Overlap avoids severing a sentence right at a chunk boundary from the
    context that would make it findable -- a chunk ending mid-thought loses
    the concept it was building toward, which is exactly the kind of thing a
    later semantic search on that concept needs to still catch.
    """
    chunks: list[Chunk] = []
    for block in blocks:
        text = block.text
        if len(text) <= chunk_chars:
            chunks.append(Chunk(section=block.section, text=text))
            continue
        start = 0
        while start < len(text):
            end = min(start + chunk_chars, len(text))
            chunks.append(Chunk(section=block.section, text=text[start:end]))
            if end >= len(text):
                break
            start = end - overlap_chars
    return chunks
