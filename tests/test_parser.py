from pathlib import Path

import pytest

from library_mcp.parser import ParseError, TextBlock, chunk_blocks, detect_doc_type, extract

_FIXTURES = Path(__file__).parent / "fixtures"


def test_chunk_blocks_splits_long_text_with_overlap() -> None:
    blocks = [TextBlock(section="p1", text="x" * 3000)]
    chunks = chunk_blocks(blocks, chunk_chars=1200, overlap_chars=150)

    assert len(chunks) == 3
    assert all(c.section == "p1" for c in chunks)
    assert len(chunks[0].text) == 1200
    assert len(chunks[-1].text) == 900


def test_chunk_blocks_leaves_short_text_whole() -> None:
    blocks = [TextBlock(section="p1", text="short")]
    chunks = chunk_blocks(blocks, chunk_chars=1200, overlap_chars=150)

    assert len(chunks) == 1
    assert chunks[0].text == "short"


def test_extract_pdf_real_file() -> None:
    blocks = extract(_FIXTURES / "networking-with-python.pdf")

    assert len(blocks) == 2
    assert "sockets" in blocks[0].text.lower()


def test_extract_epub_excludes_nav_document() -> None:
    blocks = extract(_FIXTURES / "python-fundamentals.epub")

    sections = [b.section for b in blocks]
    assert "nav.xhtml" not in sections
    assert len(blocks) == 2


def test_extract_unsupported_extension_raises(tmp_path: Path) -> None:
    bogus = tmp_path / "book.txt"
    bogus.write_text("hello")

    with pytest.raises(ParseError):
        extract(bogus)


def test_extract_title_returns_none_for_reportlab_default() -> None:
    # A real, verified finding: reportlab's default PDF title is literally
    # the placeholder string "untitled" when nothing sets it explicitly --
    # worse than the filename-derived fallback the caller already has.
    from library_mcp.parser import extract_title

    assert extract_title(_FIXTURES / "networking-with-python.pdf") is None


def test_extract_title_returns_real_metadata_title() -> None:
    from library_mcp.parser import extract_title

    assert extract_title(_FIXTURES / "titled-book.pdf") == "A Real Book Title"


def test_extract_title_returns_epub_dc_title() -> None:
    from library_mcp.parser import extract_title

    assert extract_title(_FIXTURES / "python-fundamentals.epub") == "Python Fundamentals"


def test_detect_doc_type_epub_is_always_book() -> None:
    # No article/notes case ever applies to EPUB -- verified default, not a guess.
    blocks = [TextBlock(section="c1", text="Abstract\nKeywords: x\nReferences\n10.1234/abcd")]
    assert detect_doc_type(blocks, ".epub") == "book"


def test_detect_doc_type_recognizes_a_scientific_article() -> None:
    blocks = [
        TextBlock(
            section="p1",
            text=(
                "Abstract\nThis paper studies X.\nKeywords: networking, latency\n\n"
                + ("Body text discussing the method in detail. " * 40)
            ),
        ),
        TextBlock(
            section="p2",
            text=(
                ("More body text about results and discussion. " * 40)
                + "\nReferences\n[1] Smith, J. (2020). doi:10.1234/abcd5678"
            ),
        ),
    ]
    assert detect_doc_type(blocks, ".pdf") == "article"


def test_detect_doc_type_recognizes_short_notes() -> None:
    blocks = [TextBlock(section=None, text="Buy milk. Call Alice back. Renew car insurance.")]
    assert detect_doc_type(blocks, ".pdf") == "notes"


def test_detect_doc_type_defaults_to_book() -> None:
    # A long PDF with no article markers and no notes-length text -- the
    # existing 11 real ingested books are exactly this shape.
    blocks = [
        TextBlock(section=f"page {i}", text="Chapter body text. " * 200) for i in range(1, 12)
    ]
    assert detect_doc_type(blocks, ".pdf") == "book"


def test_detect_doc_type_single_article_signal_is_not_enough() -> None:
    # Guards against over-eager classification -- a book that happens to
    # mention "references" once (e.g. a bibliography chapter title) should
    # not flip to "article" on that alone.
    blocks = [
        TextBlock(section=f"page {i}", text="Chapter body text. " * 200) for i in range(1, 8)
    ] + [TextBlock(section="page 8", text="References\nSee the bibliography chapter.")]
    assert detect_doc_type(blocks, ".pdf") == "book"


def test_detect_doc_type_a_real_academic_book_is_not_an_article() -> None:
    # Real bug caught live: a full-length academic book with an ordinary
    # bibliography (containing a DOI, since that's completely normal citation
    # practice) and its own "References" section at the end used to hit the
    # 2-signal threshold (has_doi + has_references) and get misclassified as
    # "article" -- even though it's hundreds of pages long. The length
    # ceiling and restricting DOI-search to the head (a real article states
    # its own DOI up front, not buried in a citation) both fix this.
    long_body = "Chapter body text discussing the subject at length. " * 3000
    blocks = [
        TextBlock(section="front matter", text="A Book About Something\nBy An Author"),
        TextBlock(section="body", text=long_body),
        TextBlock(
            section="back matter",
            text="References\n[1] Smith, J. (2020). Some Paper. doi:10.1234/abcd5678",
        ),
    ]
    assert detect_doc_type(blocks, ".pdf") == "book"
