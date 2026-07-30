"""Schema migration + storage-layer tests for book structuring
(docs/planning/book-structuring.md §6, §9 build step 0)."""

from pathlib import Path

from library_mcp.store import (
    add_book,
    add_chapter,
    add_chunk,
    add_glossary_term,
    book_structuring_status,
    commit,
    list_chapters_for_book,
    list_epub_books_needing_structuring,
    list_sections_for_book,
    lookup_glossary_term,
    mark_chapter_status,
    open_store,
)


def test_open_store_creates_the_book_structuring_tables(tmp_path: Path) -> None:
    conn = open_store(tmp_path / "test.db")
    book_id = add_book(conn, "A Book", "/inbox/a.epub", "2026-01-01")
    chapter_id = add_chapter(conn, book_id, 0, "chap1.xhtml")
    add_glossary_term(
        conn,
        book_id,
        chapter_id,
        "Recursion",
        "A function calling itself.",
        "deepseek-r1:8b",
        "2026-07-30T00:00:00",
    )
    chapters = list_chapters_for_book(conn, book_id)
    assert len(chapters) == 1
    assert chapters[0].status == "pending"
    entry = lookup_glossary_term(conn, "Recursion")
    assert entry is not None
    assert entry.book_title == "A Book"


def test_open_store_migrates_a_pre_book_structuring_database(tmp_path: Path) -> None:
    # Simulates the real knowledge.db before this feature: books/chunks
    # exist, book_chapters/book_glossary_terms do not.
    import sqlite3

    db_path = tmp_path / "old.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE books (
            id INTEGER PRIMARY KEY, title TEXT NOT NULL, source_path TEXT NOT NULL,
            ingested_at TEXT NOT NULL
        );
        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY, book_id INTEGER NOT NULL, chunk_index INTEGER NOT NULL,
            section TEXT, text TEXT NOT NULL, embedding TEXT NOT NULL
        );
        """
    )
    conn.execute(
        "INSERT INTO books (id, title, source_path, ingested_at) VALUES (1, 'Old Book', "
        "'/inbox/old.epub', '2026-01-01')"
    )
    conn.commit()
    conn.close()

    reopened = open_store(db_path)  # must not error
    chapter_id = add_chapter(reopened, 1, 0, "chap1.xhtml")
    assert chapter_id > 0


def test_add_chapter_defaults_to_pending() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        conn = open_store(Path(td) / "test.db")
        book_id = add_book(conn, "A Book", "/inbox/a.epub", "2026-01-01")
        chapter_id = add_chapter(conn, book_id, 0, "chap1.xhtml")
        chapters = list_chapters_for_book(conn, book_id)
        assert chapters[0].id == chapter_id
        assert chapters[0].status == "pending"
        assert chapters[0].model is None
        assert chapters[0].structured_at is None


def test_mark_chapter_status_sets_model_and_timestamp_without_clobbering(tmp_path: Path) -> None:
    conn = open_store(tmp_path / "test.db")
    book_id = add_book(conn, "A Book", "/inbox/a.epub", "2026-01-01")
    chapter_id = add_chapter(conn, book_id, 0, "chap1.xhtml")

    mark_chapter_status(
        conn, chapter_id, "done", model="deepseek-r1:8b", structured_at="2026-07-30T00:00:00"
    )
    chapter = list_chapters_for_book(conn, book_id)[0]
    assert chapter.status == "done"
    assert chapter.model == "deepseek-r1:8b"
    assert chapter.structured_at == "2026-07-30T00:00:00"

    # A later status change with no new model/timestamp keeps the old ones
    # (COALESCE) -- e.g. re-marking failed after retry logic doesn't erase
    # provenance from an earlier successful attempt.
    mark_chapter_status(conn, chapter_id, "failed")
    chapter = list_chapters_for_book(conn, book_id)[0]
    assert chapter.status == "failed"
    assert chapter.model == "deepseek-r1:8b"
    assert chapter.structured_at == "2026-07-30T00:00:00"


def test_list_chapters_for_book_orders_by_chapter_index(tmp_path: Path) -> None:
    conn = open_store(tmp_path / "test.db")
    book_id = add_book(conn, "A Book", "/inbox/a.epub", "2026-01-01")
    add_chapter(conn, book_id, 2, "chap3.xhtml")
    add_chapter(conn, book_id, 0, "chap1.xhtml")
    add_chapter(conn, book_id, 1, "chap2.xhtml")

    chapters = list_chapters_for_book(conn, book_id)
    assert [c.chapter_index for c in chapters] == [0, 1, 2]


def test_list_sections_for_book_reconstructs_chapter_blocks_from_chunks(tmp_path: Path) -> None:
    # Same section-grouping shape store.search()'s neighbor-expansion relies
    # on: multiple chunk_index rows sharing one section == one chapter.
    conn = open_store(tmp_path / "test.db")
    book_id = add_book(conn, "A Book", "/inbox/a.epub", "2026-01-01")
    add_chunk(conn, book_id, 0, "chap1.xhtml", "Intro to async.", [1.0, 0.0])
    add_chunk(conn, book_id, 1, "chap1.xhtml", "More on async.", [1.0, 0.0])
    add_chunk(conn, book_id, 2, "chap2.xhtml", "Syntax basics.", [0.0, 1.0])
    commit(conn)

    sections = list_sections_for_book(conn, book_id)
    assert [s for s, _ in sections] == ["chap1.xhtml", "chap2.xhtml"]
    assert sections[0][1] == "Intro to async.\nMore on async."
    assert sections[1][1] == "Syntax basics."


def test_list_sections_for_book_ignores_null_section_chunks(tmp_path: Path) -> None:
    # learn_text books have section=NULL for every chunk (§1.1) -- these
    # must never surface as a "chapter" with a None section.
    conn = open_store(tmp_path / "test.db")
    book_id = add_book(conn, "Learned Text", "https://example.com/article", "2026-01-01")
    add_chunk(conn, book_id, 0, None, "some pasted text", [1.0, 0.0])
    commit(conn)

    assert list_sections_for_book(conn, book_id) == []


def test_list_epub_books_needing_structuring_selects_only_epub_and_unstructured(
    tmp_path: Path,
) -> None:
    conn = open_store(tmp_path / "test.db")
    epub_id = add_book(conn, "Epub Book", "/inbox/a.epub", "2026-01-01")
    add_book(conn, "PDF Book", "/inbox/b.pdf", "2026-01-02")
    add_book(conn, "Learned Text", "https://example.com/x", "2026-01-03")
    commit(conn)

    candidates = list_epub_books_needing_structuring(conn, limit=10)
    assert [c.book_id for c in candidates] == [epub_id]


def test_list_epub_books_needing_structuring_is_oldest_first(tmp_path: Path) -> None:
    conn = open_store(tmp_path / "test.db")
    newer_id = add_book(conn, "Newer", "/inbox/newer.epub", "2026-01-05")
    older_id = add_book(conn, "Older", "/inbox/older.epub", "2026-01-01")
    commit(conn)

    candidates = list_epub_books_needing_structuring(conn, limit=10)
    assert [c.book_id for c in candidates] == [older_id, newer_id]


def test_list_epub_books_needing_structuring_excludes_fully_done_books(tmp_path: Path) -> None:
    conn = open_store(tmp_path / "test.db")
    book_id = add_book(conn, "Done Book", "/inbox/a.epub", "2026-01-01")
    chapter_id = add_chapter(conn, book_id, 0, "chap1.xhtml")
    mark_chapter_status(
        conn, chapter_id, "done", model="deepseek-r1:8b", structured_at="2026-07-30T00:00:00"
    )

    assert list_epub_books_needing_structuring(conn, limit=10) == []


def test_list_epub_books_needing_structuring_reselects_a_partially_done_book(
    tmp_path: Path,
) -> None:
    # A book with SOME chapters done and at least one still pending (e.g. a
    # crashed prior run) must be re-selected so it's resumable, not stuck
    # forever (§6's "partial progress is visible and resumable").
    conn = open_store(tmp_path / "test.db")
    book_id = add_book(conn, "Partial Book", "/inbox/a.epub", "2026-01-01")
    done_id = add_chapter(conn, book_id, 0, "chap1.xhtml")
    mark_chapter_status(
        conn, done_id, "done", model="deepseek-r1:8b", structured_at="2026-07-30T00:00:00"
    )
    add_chapter(conn, book_id, 1, "chap2.xhtml")  # still pending

    candidates = list_epub_books_needing_structuring(conn, limit=10)
    assert [c.book_id for c in candidates] == [book_id]


def test_list_epub_books_needing_structuring_respects_limit(tmp_path: Path) -> None:
    conn = open_store(tmp_path / "test.db")
    add_book(conn, "One", "/inbox/one.epub", "2026-01-01")
    add_book(conn, "Two", "/inbox/two.epub", "2026-01-02")
    add_book(conn, "Three", "/inbox/three.epub", "2026-01-03")
    commit(conn)

    candidates = list_epub_books_needing_structuring(conn, limit=2)
    assert len(candidates) == 2


def test_lookup_glossary_term_exact_match_wins_over_substring(tmp_path: Path) -> None:
    conn = open_store(tmp_path / "test.db")
    book_id = add_book(conn, "A Book", "/inbox/a.epub", "2026-01-01")
    chapter_id = add_chapter(conn, book_id, 0, "chap1.xhtml")
    add_glossary_term(
        conn,
        book_id,
        chapter_id,
        "recursion theorem",
        "A longer, unrelated theorem name.",
        "deepseek-r1:8b",
        "2026-07-30T00:00:00",
    )
    add_glossary_term(
        conn,
        book_id,
        chapter_id,
        "recursion",
        "A function that calls itself.",
        "deepseek-r1:8b",
        "2026-07-30T00:01:00",
    )

    entry = lookup_glossary_term(conn, "recursion")
    assert entry is not None
    assert entry.term == "recursion"
    assert entry.definition == "A function that calls itself."


def test_lookup_glossary_term_is_case_insensitive(tmp_path: Path) -> None:
    conn = open_store(tmp_path / "test.db")
    book_id = add_book(conn, "A Book", "/inbox/a.epub", "2026-01-01")
    chapter_id = add_chapter(conn, book_id, 0, "chap1.xhtml")
    add_glossary_term(
        conn,
        book_id,
        chapter_id,
        "Recursion",
        "A function calling itself.",
        "deepseek-r1:8b",
        "2026-07-30T00:00:00",
    )

    assert lookup_glossary_term(conn, "recursion") is not None
    assert lookup_glossary_term(conn, "RECURSION") is not None


def test_lookup_glossary_term_falls_back_to_substring_match(tmp_path: Path) -> None:
    conn = open_store(tmp_path / "test.db")
    book_id = add_book(conn, "A Book", "/inbox/a.epub", "2026-01-01")
    chapter_id = add_chapter(conn, book_id, 0, "chap1.xhtml")
    add_glossary_term(
        conn,
        book_id,
        chapter_id,
        "tail recursion",
        "Recursion where the recursive call is the last action.",
        "deepseek-r1:8b",
        "2026-07-30T00:00:00",
    )

    entry = lookup_glossary_term(conn, "recursion")
    assert entry is not None
    assert entry.term == "tail recursion"


def test_lookup_glossary_term_returns_none_when_no_match(tmp_path: Path) -> None:
    conn = open_store(tmp_path / "test.db")
    assert lookup_glossary_term(conn, "quokka") is None


def test_book_structuring_status_reports_no_for_never_queued_book(tmp_path: Path) -> None:
    conn = open_store(tmp_path / "test.db")
    add_book(conn, "PDF Book", "/inbox/a.pdf", "2026-01-01")

    statuses = book_structuring_status(conn)
    assert statuses[0].title == "PDF Book"
    assert statuses[0].structured == "no"


def test_book_structuring_status_reports_pending_and_yes(tmp_path: Path) -> None:
    conn = open_store(tmp_path / "test.db")
    pending_book = add_book(conn, "Pending Book", "/inbox/a.epub", "2026-01-01")
    add_chapter(conn, pending_book, 0, "chap1.xhtml")

    done_book = add_book(conn, "Done Book", "/inbox/b.epub", "2026-01-02")
    chapter_id = add_chapter(conn, done_book, 0, "chap1.xhtml")
    mark_chapter_status(
        conn, chapter_id, "done", model="deepseek-r1:8b", structured_at="2026-07-30T00:00:00"
    )

    statuses = {s.title: s.structured for s in book_structuring_status(conn)}
    assert statuses["Pending Book"] == "pending"
    assert statuses["Done Book"] == "yes"
