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
    # The block's real, human-readable title when the source document
    # actually declares one (EPUB `book.toc`) -- additive and optional:
    # `section` keeps its existing contract as the stable, unique grouping
    # key (a spine-item filename for EPUB, "page N" for PDF), because two
    # spine items can legitimately carry the same TOC title and merging
    # them would silently fuse two chapters into one.
    title: str | None = None


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
_MAX_TITLE_CHARS = 300


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
    if not cleaned or len(cleaned) > _MAX_TITLE_CHARS:
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
            # Lazy, not top-level: mirrors extract_epub's own ImportError ->
            # ParseError conversion below, so a broken ebooklib install fails
            # gracefully here too instead of crashing the whole module at
            # import time.
            from ebooklib import epub  # noqa: PLC0415

            book = epub.read_epub(str(path))
            values = book.get_metadata("DC", "title")
            if values:
                return _clean_title(values[0][0])
            return None
    except Exception:
        return None
    return None


# ---------------------------------------------------------------------------
# Project Gutenberg boilerplate
# ---------------------------------------------------------------------------
#
# Real incident, 2026-07-30: book-structuring's first real run extracted a
# 6-term glossary for Spinoza's "On the Improvement of the Understanding"
# (Gutenberg EPUB pg1016) in which every single term -- Defects, Damages,
# Disclaimer Of Warranties, Right Of Replacement Or Refund, Indemnity --
# came from Gutenberg's own terms-of-use text, not from Spinoza. The
# license ships inside the SAME spine item as the closing pages of the real
# book, so neither the nav/TOC exclusion below nor any bare length filter
# could catch it: it is long, prose-shaped, and structurally
# indistinguishable from a chapter.
#
# Deterministic, not model-judged -- this stack's house rule
# ([[pythia-model-tool-calling]]): Gutenberg's wrapper is a standardized
# template, verified stable across 24 real Gutenberg EPUBs in this project's
# own live library (books 16-104 in ~/.pythia/library/data/knowledge.db).
# All 24 carry the `*** START OF THE PROJECT GUTENBERG EBOOK <TITLE> ***`
# banner verbatim; 20 also carry the matching `*** END OF ... ***` banner
# followed by the full license.
#
# Two complementary functions, deliberately:
#   `strip_gutenberg_boilerplate` trims the wrapper off a block that also
#   holds real book text (the Spinoza case) -- never drops the whole block,
#   because the real content and the license live together.
#   `is_gutenberg_boilerplate` decides whether what's left is *only*
#   license, for the other real layout where Gutenberg's terms are their
#   own separate spine item.
#
# Matching is whitespace-normalized: verified live that the extracted text
# wraps "Right of Replacement or Refund" across a newline, so a naive
# substring test on the raw text misses it.

_GUTENBERG_START_RE = re.compile(
    r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
    re.IGNORECASE | re.DOTALL,
)
_GUTENBERG_END_RE = re.compile(
    r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
    re.IGNORECASE | re.DOTALL,
)

# Phrases from Gutenberg's standard license template, each verified present
# verbatim (after whitespace normalization) in this library's real Gutenberg
# EPUBs. Conjunctive by design, in the same spirit as
# reflex/zero_tool_fabrication.py's claim patterns: no single one of these
# fires on its own, because a real book about copyright or a real book that
# merely mentions Project Gutenberg must never be mistaken for the license.
_GUTENBERG_LICENSE_MARKERS = (
    "section 1. general terms of use",
    "www.gutenberg.org/license",
    "right of replacement or refund",
    "the full project gutenberg license",
    "project gutenberg literary archive foundation",
    "project gutenberg™ electronic works",
    "project gutenberg-tm electronic works",
    "start: full license",
)
_GUTENBERG_MIN_MARKERS = 3
# A block can only be *entirely* boilerplate if the license makes up
# essentially all of it. Gutenberg's full license is ~18k characters; a real
# chapter that happens to quote a few license lines is not boilerplate.
_GUTENBERG_MAX_RESIDUAL_CHARS = 2000

_NORMALIZE_WS_RE = re.compile(r"\s+")


def _normalized(text: str) -> str:
    return _NORMALIZE_WS_RE.sub(" ", text).strip().lower()


def _gutenberg_marker_count(text: str) -> int:
    normalized = _normalized(text)
    return sum(1 for marker in _GUTENBERG_LICENSE_MARKERS if marker in normalized)


def strip_gutenberg_boilerplate(text: str) -> str:
    """Trim Project Gutenberg's own wrapper off a block of real book text.

    Everything before the `*** START OF ... ***` banner (Gutenberg's header:
    title/author/release-date/encoding block) and everything from the
    `*** END OF ... ***` banner onward (the full terms-of-use license) is
    removed. Text between the banners -- the actual book -- is returned
    untouched, and a block with no banners at all is returned verbatim.

    Never drops content on suspicion: only these two exact, standardized
    banners move the boundaries.
    """
    start = _GUTENBERG_START_RE.search(text)
    if start is not None:
        text = text[start.end() :]
    end = _GUTENBERG_END_RE.search(text)
    if end is not None:
        text = text[: end.start()]
    return text.strip()


def is_gutenberg_boilerplate(text: str) -> bool:
    """True when a block is Gutenberg's legal boilerplate rather than book text.

    Requires several independent license markers AND that almost nothing is
    left once the license body is accounted for, so a chapter that merely
    quotes or discusses the license is kept.
    """
    if _gutenberg_marker_count(text) < _GUTENBERG_MIN_MARKERS:
        return False
    # Where does the license actually begin? Everything ahead of the first
    # marker is candidate real content; if there's a meaningful amount of
    # it, this block is a mixed block for strip_gutenberg_boilerplate to
    # handle, not a pure-boilerplate block to discard.
    normalized = _normalized(text)
    first = min(
        (normalized.find(marker) for marker in _GUTENBERG_LICENSE_MARKERS if marker in normalized),
        default=0,
    )
    return first <= _GUTENBERG_MAX_RESIDUAL_CHARS


def _clean_block_text(text: str) -> str | None:
    """Shared Gutenberg filter for both extractors: trim the wrapper, then
    drop the block entirely if what remains is only license text (or
    nothing). Returns None when the block should be skipped."""
    cleaned = strip_gutenberg_boilerplate(text)
    if not cleaned or is_gutenberg_boilerplate(cleaned):
        return None
    return cleaned


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
        if not text:
            continue
        # Gutenberg distributes PDFs carrying the exact same standardized
        # header/license wrapper as its EPUBs, so a PDF page gets the same
        # treatment: trim the banners, drop a page that is only license.
        cleaned = _clean_block_text(text)
        if cleaned is None:
            continue
        blocks.append(TextBlock(section=f"page {i + 1}", text=cleaned))
    if not blocks:
        msg = (
            f"{path.name}: no extractable text "
            "(likely a scanned/image-only PDF -- OCR not supported)"
        )
        raise ParseError(msg)
    return blocks


def _toc_titles(toc: object) -> dict[str, str]:
    """Flatten `ebooklib`'s parsed table of contents into {item name: title}.

    `book.toc` is a real, separate API from the `EpubNav` *content* item
    excluded from chunks below: it is a Python structure of `epub.Link`
    (href/title/uid) and `(epub.Section, [children])` tuples, nested
    arbitrarily deep, built from the EPUB's nav/NCX at read time -- verified
    directly against ebooklib's installed source and against a real
    Gutenberg EPUB in this library, whose hrefs match spine-item names
    exactly once the `#fragment` is dropped.

    First entry per item wins: several TOC entries routinely point into the
    same spine item at different anchors, and the first is the one whose
    heading actually opens that file.
    """
    titles: dict[str, str] = {}

    def visit(entries: object) -> None:
        if not isinstance(entries, (list, tuple)):
            return
        for entry in entries:
            if isinstance(entry, tuple):
                # (Section, children) -- the Section itself carries a real
                # href/title too, so record it before descending.
                if entry:
                    visit([entry[0]])
                if len(entry) > 1:
                    visit(entry[1])
                continue
            href = getattr(entry, "href", None)
            title = getattr(entry, "title", None)
            if not href or not title:
                continue
            name = str(href).split("#", 1)[0]
            if name and name not in titles:
                titles[name] = str(title).strip()

    visit(toc)
    return titles


def extract_epub(path: Path) -> list[TextBlock]:
    try:
        import ebooklib  # noqa: PLC0415
        from ebooklib import epub  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - import-time dependency check
        msg = "ebooklib is not installed"
        raise ParseError(msg) from exc
    try:
        book = epub.read_epub(str(path))
    except Exception as exc:
        msg = f"could not open EPUB {path.name}: {exc}"
        raise ParseError(msg) from exc
    blocks: list[TextBlock] = []
    # Leo's ask: "if the book has a table of contents, use it when needed."
    # Purely additive -- an empty/absent/incomplete TOC leaves every block's
    # title None and the filename-based `section` label unchanged, which is
    # exactly the pre-existing behaviour downstream already relies on.
    toc_titles = _toc_titles(getattr(book, "toc", None))
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
        if not text:
            continue
        # Project Gutenberg's header/license wrapper is not book content --
        # same reasoning as the nav/TOC exclusion above, one step further:
        # the license is prose-shaped and long enough to look like a real
        # chapter to everything downstream. Verified live 2026-07-30, where
        # it produced an entire glossary ("Defects", "Indemnity", ...) for a
        # Spinoza book. See the block comment above extract_pdf.
        cleaned = _clean_block_text(text)
        if cleaned is None:
            continue
        name = item.get_name()
        blocks.append(TextBlock(section=name, text=cleaned, title=toc_titles.get(name)))
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


_ABSTRACT_RE = re.compile(r"\babstract\b", re.IGNORECASE)
_DOI_RE = re.compile(r"\bdoi\s*[:\-]?\s*10\.\d{4,9}/", re.IGNORECASE)
_KEYWORDS_RE = re.compile(r"\bkeywords\s*[:\-]", re.IGNORECASE)
_REFERENCES_RE = re.compile(r"\b(references|bibliography)\b", re.IGNORECASE)
_NOTES_MAX_BLOCKS = 6
_NOTES_MAX_CHARS = 4000
_ARTICLE_MIN_SIGNALS = 2
# A real scientific article is short -- a document this long is a book
# (or thesis) even if it has a DOI/references section somewhere in it.
# Found live: a bibliography containing one DOI, or ending in a
# "References" section (both completely ordinary in a real book), was
# enough on its own to misclassify a full-length academic book as an
# "article" before this ceiling existed.
_ARTICLE_MAX_CHARS = 150_000


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
    if total_chars > _ARTICLE_MAX_CHARS:
        return "book"
    lower = full_text.lower()
    head = lower[:2000]
    tail = lower[-4000:]

    has_abstract = _ABSTRACT_RE.search(head) is not None
    has_keywords = _KEYWORDS_RE.search(head) is not None
    # DOI restricted to the head, same reasoning as the length ceiling above:
    # a real article states its own DOI up front, on page one -- one buried
    # in a bibliography entry just means the book cites someone else's paper.
    has_doi = _DOI_RE.search(head) is not None
    has_references = _REFERENCES_RE.search(tail) is not None

    article_signals = sum([has_abstract, has_keywords, has_doi, has_references])
    if article_signals >= _ARTICLE_MIN_SIGNALS:
        return "article"

    if len(blocks) <= _NOTES_MAX_BLOCKS and total_chars < _NOTES_MAX_CHARS and article_signals == 0:
        return "notes"

    return "book"


@dataclass(frozen=True, slots=True)
class Chunk:
    section: str | None
    text: str
    # Carried straight through from the block's TextBlock.title (the EPUB's
    # own TOC entry for this spine item), so the ingestion path can persist
    # a real chapter title alongside the filename-shaped `section` key.
    title: str | None = None


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
            chunks.append(Chunk(section=block.section, text=text, title=block.title))
            continue
        start = 0
        while start < len(text):
            end = min(start + chunk_chars, len(text))
            chunks.append(Chunk(section=block.section, text=text[start:end], title=block.title))
            if end >= len(text):
                break
            start = end - overlap_chars
    return chunks
