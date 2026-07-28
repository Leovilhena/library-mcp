import asyncio
from pathlib import Path
from typing import Callable

_FIXTURES = Path(__file__).parent / "fixtures"


def _generate_fixtures() -> None:
    """Create the real PDF/EPUB test fixtures if they're not already present.

    These are gitignored (real binary files, not source) -- a fresh clone of
    this repo has none of them, and every test importing this module used to
    rely on them existing anyway, only working because they happened to
    still be sitting on disk from a manual generation step earlier in this
    project. Found live, 2026-07-28, while adding a fourth fixture by hand
    the same way. Generated once per test session, idempotent (skipped if
    already present).
    """
    _FIXTURES.mkdir(exist_ok=True)

    pdf_path = _FIXTURES / "networking-with-python.pdf"
    if not pdf_path.exists():
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas

        c = canvas.Canvas(str(pdf_path), pagesize=letter)
        c.drawString(72, 720, "Chapter 1: Sockets")
        c.drawString(72, 700, "This chapter covers TCP sockets and how Python asyncio")
        c.drawString(72, 680, "event loops handle non-blocking network I/O efficiently.")
        c.showPage()
        c.drawString(72, 720, "Chapter 2: Handshakes")
        c.drawString(72, 700, "The TCP three-way handshake establishes a connection")
        c.drawString(72, 680, "before any application data can be exchanged.")
        c.showPage()
        c.save()

    titled_pdf_path = _FIXTURES / "titled-book.pdf"
    if not titled_pdf_path.exists():
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas

        c = canvas.Canvas(str(titled_pdf_path), pagesize=letter)
        c.setTitle("A Real Book Title")
        c.drawString(72, 720, "Some content here.")
        c.showPage()
        c.save()

    epub_path = _FIXTURES / "python-fundamentals.epub"
    if not epub_path.exists():
        from ebooklib import epub

        book = epub.EpubBook()
        book.set_identifier("test-python-book")
        book.set_title("Python Fundamentals")
        book.set_language("en")

        c1 = epub.EpubHtml(title="Async Basics", file_name="chap1.xhtml", lang="en")
        c1.content = (
            "<html><body><h1>Async Basics</h1><p>Python asyncio provides event loops "
            "and coroutines for writing concurrent code without threads.</p></body></html>"
        )
        book.add_item(c1)

        c2 = epub.EpubHtml(title="Syntax", file_name="chap2.xhtml", lang="en")
        c2.content = (
            "<html><body><h1>Syntax</h1><p>Python uses indentation to define code "
            "blocks instead of curly braces.</p></body></html>"
        )
        book.add_item(c2)

        book.toc = (c1, c2)
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())
        book.spine = ["nav", c1, c2]

        epub.write_epub(str(epub_path), book)


_generate_fixtures()


async def run_and_wait(fn: Callable[..., str], *args: object) -> str:
    """Call a `learn`/`learn_text`-style function that schedules ingestion in
    the background via `asyncio.create_task` and returns immediately, then
    wait for that background task to actually finish before the caller
    asserts on resulting DB state.
    """
    before = asyncio.all_tasks()
    result = fn(*args)
    new_tasks = asyncio.all_tasks() - before
    for task in new_tasks:
        await task
    return result
