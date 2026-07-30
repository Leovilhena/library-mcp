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


# ---------------------------------------------------------------------------
# Project Gutenberg boilerplate (real incident, 2026-07-30)
# ---------------------------------------------------------------------------

# A real, verbatim clean chunk from a genuinely ingested Gutenberg book in
# this project's live library (book_id 97, "Collected Papers on Analytical
# Psychology") -- the negative fixture has to be real book prose from a real
# Gutenberg release, because that is exactly the text the detector must not
# touch.
_REAL_BOOK_PROSE = (
    "The dream is analysed by rumour--Psychoanalysis explains the construction of "
    "rumour--The dream gives the watchword for the unconscious--It brings to "
    "expression the ready-prepared sexual complexes--Marie X.'s unsatisfactory "
    "conduct brought her under reproof--Her indignation and repressed feelings lead "
    "to the dream--She uses this as an instrument of revenge against the teacher. "
    "CHAPTER V On the Significance of Number-Dreams. Symbolism of numbers has "
    "acquired fresh interest from Freud's investigations."
)


def test_is_gutenberg_boilerplate_detects_the_real_license() -> None:
    from conftest import GUTENBERG_LICENSE_TEXT
    from library_mcp.parser import is_gutenberg_boilerplate

    assert is_gutenberg_boilerplate(GUTENBERG_LICENSE_TEXT)


def test_is_gutenberg_boilerplate_matches_across_line_wrapping() -> None:
    # Verified live: the extracted text wraps "Right of Replacement or
    # Refund" across a newline, so a naive substring test on raw text misses
    # it. Matching is whitespace-normalized precisely because of this.
    from library_mcp.parser import is_gutenberg_boilerplate

    wrapped = (
        "1.F.3. LIMITED RIGHT OF\nREPLACEMENT OR REFUND\n"
        "available with this file or online at\nwww.gutenberg.org/license.\n"
        "Section 1. General\nTerms of Use and Redistributing"
    )
    assert is_gutenberg_boilerplate(wrapped)


def test_is_gutenberg_boilerplate_ignores_real_book_prose() -> None:
    from library_mcp.parser import is_gutenberg_boilerplate

    assert not is_gutenberg_boilerplate(_REAL_BOOK_PROSE)


def test_is_gutenberg_boilerplate_needs_more_than_one_marker() -> None:
    # Conjunctive by design: a real book that merely mentions Project
    # Gutenberg, or one whose own subject is copyright law, must not be
    # mistaken for the license.
    from library_mcp.parser import is_gutenberg_boilerplate

    passing_mention = (
        "This edition was prepared from the Project Gutenberg Literary Archive "
        "Foundation's scan, and the author discusses copyright at length in the "
        "chapters that follow. " + _REAL_BOOK_PROSE
    )
    assert not is_gutenberg_boilerplate(passing_mention)


def test_strip_gutenberg_boilerplate_keeps_the_book_and_drops_the_wrapper() -> None:
    from conftest import GUTENBERG_LICENSE_TEXT
    from library_mcp.parser import strip_gutenberg_boilerplate

    mixed = (
        "The Project Gutenberg eBook of Something\nRelease date: 1997\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK SOMETHING ***\n"
        + _REAL_BOOK_PROSE
        + "\n*** END OF THE PROJECT GUTENBERG EBOOK SOMETHING ***\n"
        + GUTENBERG_LICENSE_TEXT
    )
    cleaned = strip_gutenberg_boilerplate(mixed)

    assert cleaned == _REAL_BOOK_PROSE
    assert "Release date" not in cleaned
    assert "FULL LICENSE" not in cleaned


def test_strip_gutenberg_boilerplate_leaves_a_non_gutenberg_block_alone() -> None:
    from library_mcp.parser import strip_gutenberg_boilerplate

    assert strip_gutenberg_boilerplate(_REAL_BOOK_PROSE) == _REAL_BOOK_PROSE


def test_extract_epub_strips_gutenberg_wrapper_and_drops_license_section() -> None:
    # End-to-end against a fixture built in the real layout of pg1016: the
    # exact shape that produced a 6-term glossary of "Defects", "Indemnity"
    # and friends for a Spinoza book on 2026-07-30.
    blocks = extract(_FIXTURES / "gutenberg-boilerplate.epub")

    sections = [b.section for b in blocks]
    assert "pg-h-1.xhtml" not in sections, "the license-only spine item must be dropped entirely"
    assert len(blocks) == 1

    body = blocks[0].text
    assert "all the usual surroundings of social life are vain and futile" in body
    assert "Indemnity" not in body
    assert "gutenberg.org/license" not in body
    assert "Release date" not in body


def test_extract_epub_uses_the_real_table_of_contents_title() -> None:
    # Leo's ask: use the book's own TOC when it has one. `section` keeps its
    # filename identity (the stable grouping key); `title` is the additive
    # human-readable label.
    blocks = extract(_FIXTURES / "python-fundamentals.epub")

    assert [b.section for b in blocks] == ["chap1.xhtml", "chap2.xhtml"]
    assert [b.title for b in blocks] == ["Async Basics", "Syntax"]


def test_toc_titles_handles_nested_sections_and_missing_toc() -> None:
    # Real shape verified against a real Gutenberg EPUB (pg48225): a flat
    # list of Links with `#fragment` hrefs, several pointing into the same
    # spine item, plus (Section, [children]) tuples nested inside.
    from ebooklib import epub

    from library_mcp.parser import _toc_titles

    toc = [
        epub.Link("48225-h-0.htm.html#pgepubid00000", "PREFACE"),
        epub.Link("48225-h-0.htm.html#pgepubid00013", "CONTENTS"),
        (
            epub.Section("CHAPTER IX", "48225-h-8.htm.html#pgepubid00262"),
            [epub.Link("48225-h-9.htm.html#pgepubid00278", "IV")],
        ),
    ]
    titles = _toc_titles(toc)

    # First entry per spine item wins -- several TOC anchors routinely point
    # into the same file, and the first is the heading that opens it.
    assert titles["48225-h-0.htm.html"] == "PREFACE"
    assert titles["48225-h-8.htm.html"] == "CHAPTER IX"
    assert titles["48225-h-9.htm.html"] == "IV"

    # No TOC at all is a real case (the filename fallback must keep working).
    assert _toc_titles(None) == {}
    assert _toc_titles([]) == {}


def test_chunk_blocks_carries_the_toc_title_onto_every_chunk() -> None:
    blocks = [TextBlock(section="chap1.xhtml", text="x" * 3000, title="Async Basics")]
    chunks = chunk_blocks(blocks, chunk_chars=1200, overlap_chars=150)

    assert len(chunks) == 3
    assert all(c.title == "Async Basics" for c in chunks)
