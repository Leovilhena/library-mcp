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
    mark_book_status,
    open_store,
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
