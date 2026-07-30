import asyncio
from collections.abc import Callable
from pathlib import Path

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

    _generate_gutenberg_fixture()


def _generate_gutenberg_fixture() -> None:
    """A fixture in the real layout of a Project Gutenberg EPUB."""
    gutenberg_path = _FIXTURES / "gutenberg-boilerplate.epub"
    if not gutenberg_path.exists():
        from ebooklib import epub

        book = epub.EpubBook()
        book.set_identifier("test-gutenberg-book")
        book.set_title("On the Improvement of the Understanding")
        book.set_language("en")

        # Layout of a real Gutenberg EPUB, reproduced from this project's own
        # live library (pg1016, book_id 22 in ~/.pythia/library/data/
        # knowledge.db, 2026-07-30): one spine item that opens with
        # Gutenberg's header, carries the real book text, and then runs
        # straight into the END banner and the full license -- plus a second
        # spine item that is nothing but license. Both had to be handled, and
        # they need opposite treatment (trim vs. drop).
        c1 = epub.EpubHtml(title="Chapter One", file_name="pg-h-0.xhtml", lang="en")
        c1.content = (
            "<html><body>"
            "<p>The Project Gutenberg eBook of On the Improvement of the Understanding</p>"
            "<p>Release date: January 1, 1997 [eBook #1016]</p>"
            "<p>*** START OF THE PROJECT GUTENBERG EBOOK "
            "ON THE IMPROVEMENT OF THE UNDERSTANDING ***</p>"
            "<p>After experience had taught me that all the usual surroundings of social "
            "life are vain and futile, I finally resolved to inquire whether there might "
            "be some real good having power to communicate itself.</p>"
            "<p>*** END OF THE PROJECT GUTENBERG EBOOK "
            "ON THE IMPROVEMENT OF THE UNDERSTANDING ***</p>" + GUTENBERG_LICENSE_TEXT + "</body>"
            "</html>"
        )
        book.add_item(c1)

        c2 = epub.EpubHtml(title="THE FULL PROJECT GUTENBERG LICENSE", file_name="pg-h-1.xhtml")
        c2.content = "<html><body>" + GUTENBERG_LICENSE_TEXT + "</body></html>"
        book.add_item(c2)

        book.toc = (c1, c2)
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())
        book.spine = ["nav", c1, c2]

        epub.write_epub(str(gutenberg_path), book)


# Verbatim excerpt of Project Gutenberg's standard license, copied out of a
# real chunk of pg1016 in this project's live database (2026-07-30) rather
# than paraphrased -- the whole point of the detector is that it matches the
# real, standardized template, so the test data has to be the real template.
GUTENBERG_LICENSE_TEXT = """
<p>Updated editions will replace the previous one-the old editions will be renamed.</p>
<p>Creating the works from print editions not protected by U.S. copyright law means that
no one owns a United States copyright in these works, so the Foundation (and you!) can
copy and distribute it in the United States without permission and without paying
copyright royalties. Special rules, set forth in the General Terms of Use part of this
license, apply to copying and distributing Project Gutenberg-tm electronic works to
protect the PROJECT GUTENBERG-tm concept and trademark.</p>
<p>START: FULL LICENSE</p>
<p>THE FULL PROJECT GUTENBERG LICENSE</p>
<p>PLEASE READ THIS BEFORE YOU DISTRIBUTE OR USE THIS WORK</p>
<p>To protect the Project Gutenberg mission of promoting the free distribution of
electronic works, by using or distributing this work you agree to comply with all the
terms of the Full Project Gutenberg License available with this file or online at
www.gutenberg.org/license.</p>
<p>Section 1. General Terms of Use and Redistributing Project Gutenberg electronic
works</p>
<p>1.F.3. LIMITED RIGHT OF REPLACEMENT OR REFUND - If you discover a defect in this
electronic work within 90 days of receiving it, you can receive a refund of the money
(if any) you paid for it by sending a written explanation to the person you received the
work from.</p>
<p>1.F.5. Some states do not allow disclaimers of certain implied warranties or the
exclusion or limitation of certain types of damages.</p>
<p>1.F.6. INDEMNITY - You agree to indemnify and hold the Foundation, the trademark
owner, any agent or employee of the Foundation, anyone providing copies of Project
Gutenberg electronic works in accordance with this agreement, harmless from all
liability, costs and expenses, including legal fees.</p>
<p>Section 3. Information about the Project Gutenberg Literary Archive Foundation</p>
"""


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
