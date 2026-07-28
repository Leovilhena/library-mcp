from pathlib import Path

import pytest

from library_mcp.parser import ParseError, TextBlock, chunk_blocks, extract

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
