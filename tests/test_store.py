import hashlib
import sqlite3
from pathlib import Path

import pytest

from library_mcp.store import (
    DuplicateBookError,
    add_book,
    add_chunk,
    commit,
    find_book_by_hash,
    list_book_statuses,
    list_knowledge_gaps,
    mark_book_status,
    open_store,
    record_knowledge_gap,
    search,
    update_book_progress,
)


def test_search_ranks_by_similarity_across_books(tmp_path: Path) -> None:
    conn = open_store(tmp_path / "test.db")
    net_book = add_book(conn, "Networking", "/inbox/net.pdf", "2026-01-01")
    add_chunk(conn, net_book, 0, "p1", "python asyncio for networking", [1.0, 0.0, 0.0])
    add_chunk(conn, net_book, 1, "p2", "TCP handshake basics", [0.0, 1.0, 0.0])

    py_book = add_book(conn, "Python", "/inbox/py.pdf", "2026-01-01")
    add_chunk(conn, py_book, 0, "ch1", "python asyncio event loops", [0.99, 0.01, 0.0])
    commit(conn)

    results = search(conn, [1.0, 0.0, 0.0], top_k=2)

    assert len(results) == 2
    titles = {r.book_title for r in results}
    assert titles == {"Networking", "Python"}
    assert results[0].score >= results[1].score


def test_search_excludes_irrelevant_chunks_when_top_k_is_tight(tmp_path: Path) -> None:
    conn = open_store(tmp_path / "test.db")
    book = add_book(conn, "Book", "/inbox/b.pdf", "2026-01-01")
    add_chunk(conn, book, 0, "p1", "relevant", [1.0, 0.0])
    add_chunk(conn, book, 1, "p2", "irrelevant", [0.0, 1.0])
    commit(conn)

    results = search(conn, [1.0, 0.0], top_k=1)

    assert len(results) == 1
    assert results[0].text == "relevant"


def test_open_store_migrates_a_pre_dedup_database(tmp_path: Path) -> None:
    # Simulates this project's own real database on 2026-07-28: created by
    # an older schema version with no content_hash/status/progress columns,
    # and already holding a real row. open_store must not choke on it, and
    # existing rows must come out with sane defaults (a book ingested before
    # this feature existed is retroactively "done", not "embedding" forever).
    db_path = tmp_path / "old.db"
    source_file = tmp_path / "old.pdf"
    source_file.write_bytes(b"real pdf bytes")

    old_conn = sqlite3.connect(db_path)
    old_conn.executescript(
        f"""
        CREATE TABLE books (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            source_path TEXT NOT NULL,
            ingested_at TEXT NOT NULL
        );
        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY,
            book_id INTEGER NOT NULL REFERENCES books(id),
            chunk_index INTEGER NOT NULL,
            section TEXT,
            text TEXT NOT NULL,
            embedding TEXT NOT NULL
        );
        INSERT INTO books (id, title, source_path, ingested_at)
        VALUES (1, 'Old Book', '{source_file}', '2026-07-01');
        INSERT INTO chunks (book_id, chunk_index, section, text, embedding)
        VALUES (1, 0, 'p1', 'some text', '[1.0, 0.0]'),
               (1, 1, 'p2', 'more text', '[0.0, 1.0]');
        """
    )
    old_conn.commit()
    old_conn.close()

    conn = open_store(db_path)

    statuses = list_book_statuses(conn)
    assert len(statuses) == 1
    assert statuses[0].title == "Old Book"
    assert statuses[0].status == "done"
    # The actual bug caught live: without backfilling from the real chunks
    # table, this stayed 0 for every already-ingested book the moment this
    # migration ran, and list_learned showed "(0 chunks)" for real books.
    assert statuses[0].total_chunks == 2
    assert statuses[0].embedded_chunks == 2

    # The other real bug caught live: without backfilling content_hash from
    # the still-on-disk source file, re-uploading a book ingested before
    # dedup existed sailed straight past the duplicate check.
    import hashlib

    expected_hash = hashlib.sha256(source_file.read_bytes()).hexdigest()
    with pytest.raises(DuplicateBookError):
        add_book(conn, "Re-upload", "/inbox/new-name.pdf", "2026-07-02", content_hash=expected_hash)

    # doc_type backfill: a book ingested before this feature existed must
    # not be left unclassified -- "some text"/"more text" is short with no
    # article markers, so the deterministic classifier calls it "notes".
    assert statuses[0].doc_type == "notes"


def test_open_store_backfills_doc_type_as_book_for_a_real_shaped_book(tmp_path: Path) -> None:
    # A book-shaped pre-existing row (many chunks, plain prose, no article
    # markers) must backfill to "book", not "notes" -- this is the shape of
    # this project's own 11 real already-ingested books.
    db_path = tmp_path / "old.db"
    old_conn = sqlite3.connect(db_path)
    old_conn.executescript(
        """
        CREATE TABLE books (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            source_path TEXT NOT NULL,
            ingested_at TEXT NOT NULL
        );
        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY,
            book_id INTEGER NOT NULL REFERENCES books(id),
            chunk_index INTEGER NOT NULL,
            section TEXT,
            text TEXT NOT NULL,
            embedding TEXT NOT NULL
        );
        INSERT INTO books (id, title, source_path, ingested_at)
        VALUES (1, 'Real Book', '/inbox/real-book.pdf', '2026-07-01');
        """
    )
    long_text = "Chapter body text discussing the subject at length. " * 100
    for i in range(8):
        old_conn.execute(
            "INSERT INTO chunks (book_id, chunk_index, section, text, embedding) VALUES (1, ?, ?, ?, '[1.0]')",
            (i, f"page {i}", long_text),
        )
    old_conn.commit()
    old_conn.close()

    conn = open_store(db_path)

    statuses = list_book_statuses(conn)
    assert statuses[0].doc_type == "book"


def test_content_hash_backfill_retries_on_a_later_open_when_file_was_missing(tmp_path: Path) -> None:
    # Real bug caught by review: library-parse and library-keeper are two
    # separate processes sharing this db with no ordering between them, and
    # only library-parse's container can read source files. If keeper
    # happened to be the process that added the content_hash column (this
    # test simulates that by opening once while the source file is
    # unreadable), the old code's "only backfill once, in whichever process
    # added the column" gate meant a later open by a process that COULD read
    # the file would never get a second chance -- content_hash stayed NULL
    # forever, silently reintroducing the exact re-upload-sails-past-dedup
    # bug this backfill exists to prevent.
    db_path = tmp_path / "old.db"
    source_file = tmp_path / "old.pdf"

    old_conn = sqlite3.connect(db_path)
    old_conn.executescript(
        f"""
        CREATE TABLE books (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            source_path TEXT NOT NULL,
            ingested_at TEXT NOT NULL
        );
        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY,
            book_id INTEGER NOT NULL REFERENCES books(id),
            chunk_index INTEGER NOT NULL,
            section TEXT,
            text TEXT NOT NULL,
            embedding TEXT NOT NULL
        );
        INSERT INTO books (id, title, source_path, ingested_at)
        VALUES (1, 'Old Book', '{source_file}', '2026-07-01');
        """
    )
    old_conn.commit()
    old_conn.close()

    # "library-keeper" opens first -- source file not present/readable from
    # its container, so the column gets added but content_hash stays NULL.
    assert not source_file.exists()
    open_store(db_path)
    conn = open_store(db_path)
    row = conn.execute("SELECT content_hash FROM books WHERE id = 1").fetchone()
    assert row[0] is None

    # "library-parse" opens later, now that the file is actually there --
    # must still backfill, not just on the call that added the column.
    source_file.write_bytes(b"real pdf bytes")
    conn2 = open_store(db_path)
    row2 = conn2.execute("SELECT content_hash FROM books WHERE id = 1").fetchone()
    assert row2[0] is not None
    assert row2[0] == hashlib.sha256(b"real pdf bytes").hexdigest()


def test_duplicate_content_hash_is_rejected_at_the_db_level(tmp_path: Path) -> None:
    conn = open_store(tmp_path / "test.db")
    add_book(conn, "First", "/inbox/a.pdf", "2026-01-01", content_hash="abc123")

    with pytest.raises(DuplicateBookError) as exc_info:
        add_book(conn, "Second", "/inbox/b.pdf", "2026-01-01", content_hash="abc123")
    assert exc_info.value.title == "First"


def test_multiple_books_with_no_hash_do_not_collide(tmp_path: Path) -> None:
    # Partial index (WHERE content_hash IS NOT NULL) -- rows with no hash
    # must never be treated as duplicates of each other.
    conn = open_store(tmp_path / "test.db")
    add_book(conn, "First", "/inbox/a.pdf", "2026-01-01", content_hash=None)
    add_book(conn, "Second", "/inbox/b.pdf", "2026-01-01", content_hash=None)  # must not raise

    assert len(list_book_statuses(conn)) == 2


def test_find_book_by_hash(tmp_path: Path) -> None:
    conn = open_store(tmp_path / "test.db")
    add_book(conn, "Title", "/inbox/a.pdf", "2026-01-01", content_hash="deadbeef")

    assert find_book_by_hash(conn, "deadbeef") == "Title"
    assert find_book_by_hash(conn, "nonexistent") is None


def test_progress_and_status_tracking(tmp_path: Path) -> None:
    conn = open_store(tmp_path / "test.db")
    book_id = add_book(conn, "Title", "/inbox/a.pdf", "2026-01-01", status="embedding", total_chunks=10)

    update_book_progress(conn, book_id, 5)
    mid = list_book_statuses(conn)[0]
    assert mid.status == "embedding"
    assert mid.embedded_chunks == 5
    assert mid.total_chunks == 10

    mark_book_status(conn, book_id, "done")
    done = list_book_statuses(conn)[0]
    assert done.status == "done"


def test_record_knowledge_gap_inserts_a_new_row(tmp_path: Path) -> None:
    conn = open_store(tmp_path / "test.db")
    record_knowledge_gap(conn, "what does the book say about X?", "no_matches", "2026-07-28T10:00:00")

    gaps = list_knowledge_gaps(conn)
    assert len(gaps) == 1
    assert gaps[0].question == "what does the book say about X?"
    assert gaps[0].reason == "no_matches"
    assert gaps[0].times_asked == 1


def test_record_knowledge_gap_dedupes_the_same_question(tmp_path: Path) -> None:
    # Three different people asking the same unanswered question should
    # show up once, with a real repeat count -- not three separate rows.
    conn = open_store(tmp_path / "test.db")
    record_knowledge_gap(conn, "what about X?", "no_matches", "2026-07-28T10:00:00")
    record_knowledge_gap(conn, "what about X?", "no_matches", "2026-07-28T11:00:00")
    record_knowledge_gap(conn, "what about X?", "no_matches", "2026-07-28T12:00:00")

    gaps = list_knowledge_gaps(conn)
    assert len(gaps) == 1
    assert gaps[0].times_asked == 3
    assert gaps[0].first_asked_at == "2026-07-28T10:00:00"
    assert gaps[0].last_asked_at == "2026-07-28T12:00:00"


def test_list_knowledge_gaps_orders_by_times_asked(tmp_path: Path) -> None:
    conn = open_store(tmp_path / "test.db")
    record_knowledge_gap(conn, "asked once", "no_matches", "2026-07-28T10:00:00")
    record_knowledge_gap(conn, "asked twice", "no_matches", "2026-07-28T10:00:00")
    record_knowledge_gap(conn, "asked twice", "no_matches", "2026-07-28T11:00:00")

    gaps = list_knowledge_gaps(conn)
    assert [g.question for g in gaps] == ["asked twice", "asked once"]
