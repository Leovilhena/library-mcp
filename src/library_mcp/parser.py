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


_PLACEHOLDER_TITLES = {
    "untitled",
    "untitled document",
    "untitled1",
    "unknown",
    "document1",
    "new document",
}


def _clean_title(raw: str | None) -> str | None:
    """Reject a metadata title that isn't actually useful as one.

    Some PDFs/EPUBs carry a title field that's empty, just whitespace, or a
    generator's own default placeholder -- verified live: reportlab's
    default document title is literally the string "untitled" when nothing
    sets it explicitly, which would be a worse result than the
    filename-derived fallback the caller already has, not a better one.
    """
    if not raw:
        return None
    cleaned = raw.strip()
    if not cleaned or len(cleaned) > 300:
        return None
    if cleaned.lower() in _PLACEHOLDER_TITLES:
        return None
    return cleaned


def extract_title(path: Path) -> str | None:
    """Best-effort real title from the document's own metadata.

    Falls back silently (returns None) on anything that isn't a clean,
    present title -- callers already have a filename-derived fallback, and
    a numeric or otherwise meaningless filename is exactly the case this
    exists for: `learn`'s title used to be the raw filename verbatim, which
    is useless when someone uploads a book named e.g. "1234567.pdf".
    """
    suffix = path.suffix.lower()
    try:
        if suffix == ".pdf":
            reader = PdfReader(str(path))
            meta = reader.metadata
            return _clean_title(getattr(meta, "title", None) if meta else None)
        if suffix == ".epub":
            from ebooklib import epub

            book = epub.read_epub(str(path))
            values = book.get_metadata("DC", "title")
            if values:
                return _clean_title(values[0][0])
            return None
    except Exception:  # noqa: BLE001 - metadata is a nice-to-have, never worth failing ingestion over
        return None
    return None


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


_DOI_RE = re.compile(r"\bdoi\s*[:\-]?\s*10\.\d{4,9}/", re.IGNORECASE)
_KEYWORDS_RE = re.compile(r"\bkeywords\s*[:\-]", re.IGNORECASE)
_REFERENCES_RE = re.compile(r"\b(references|bibliography)\b", re.IGNORECASE)
_NOTES_MAX_BLOCKS = 6
_NOTES_MAX_CHARS = 4000
_ARTICLE_MIN_SIGNALS = 2


def detect_doc_type(blocks: list[TextBlock], suffix: str | None = None) -> str:
    """Classify a document as "book", "article", or "notes" -- deterministically.

    No LLM in the loop: this project's default model calls tools/follows
    structured-output instructions only 40-60% of the time
    ([[pythia-model-tool-calling]]), so a classification a user might rely
    on (filtering, search) needs to come from real structural markers, not
    a guess the model could get wrong or fabricate. EPUB defaults straight
    to "book" -- ebooklib's format is used here almost exclusively for
    books, never for the short-notes/scientific-article case.
    """
    if suffix == ".epub":
        return "book"
    full_text = "\n".join(b.text for b in blocks)
    total_chars = len(full_text)
    lower = full_text.lower()
    head = lower[:2000]
    tail = lower[-4000:]

    has_abstract = "abstract" in head
    has_keywords = _KEYWORDS_RE.search(head) is not None
    has_doi = _DOI_RE.search(lower) is not None
    has_references = _REFERENCES_RE.search(tail) is not None

    article_signals = sum([has_abstract, has_keywords, has_doi, has_references])
    if article_signals >= _ARTICLE_MIN_SIGNALS:
        return "article"

    if (
        len(blocks) <= _NOTES_MAX_BLOCKS
        and total_chars < _NOTES_MAX_CHARS
        and article_signals == 0
    ):
        return "notes"

    return "book"


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
