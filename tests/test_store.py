import hashlib
import inspect
import sqlite3
from pathlib import Path

import pytest

from library_mcp.store import (
    DuplicateBookError,
    add_book,
    add_chunk,
    commit,
    find_book_by_filename,
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


def test_search_pulls_in_same_section_neighbors_of_a_match(tmp_path: Path) -> None:
    # docs/TODO.md BUG-1: a real ~50-item list spanning several consecutive
    # chunk_index rows in one section, where a generic query only scores one
    # of them highly enough to rank in the raw top-k.
    conn = open_store(tmp_path / "test.db")
    book = add_book(conn, "Values Book", "/inbox/v.pdf", "2026-01-01")
    add_chunk(conn, book, 0, "ch1", "intro material, not the list", [0.0, 1.0])
    add_chunk(conn, book, 1, "values", "Acceptance, Adventure, Assertiveness", [1.0, 0.0])
    add_chunk(conn, book, 2, "values", "Courage, Curiosity, Compassion", [0.5, 0.5])
    add_chunk(conn, book, 3, "values", "Fairness, Freedom, Friendliness", [0.4, 0.6])
    add_chunk(conn, book, 4, "closing", "epilogue, not the list", [0.0, 1.0])
    commit(conn)

    # Only chunk 1 scores highly against this query; top_k=1 alone would
    # only return it and miss the rest of the same continuous list.
    results = search(conn, [1.0, 0.0], top_k=1)

    texts = {r.text for r in results}
    assert "Acceptance, Adventure, Assertiveness" in texts
    assert "Courage, Curiosity, Compassion" in texts, (
        "same-section neighbor (chunk_index 2) was not pulled in"
    )
    assert "Fairness, Freedom, Friendliness" not in texts, (
        "chunk_index 3 is not an immediate neighbor of the chunk_index 1 "
        "match and should not be pulled in by one hop"
    )
    assert "intro material, not the list" not in texts, (
        "chunk_index 0 is an immediate neighbor by index but was never a "
        "match, so it should not appear on its own"
    )


def test_search_neighbor_expansion_never_crosses_a_section_boundary(tmp_path: Path) -> None:
    conn = open_store(tmp_path / "test.db")
    book = add_book(conn, "Book", "/inbox/b.pdf", "2026-01-01")
    add_chunk(conn, book, 0, "chapter1", "end of chapter one", [1.0, 0.0])
    add_chunk(conn, book, 1, "chapter2", "start of chapter two", [0.0, 1.0])
    commit(conn)

    results = search(conn, [1.0, 0.0], top_k=1)

    texts = {r.text for r in results}
    assert texts == {"end of chapter one"}, (
        "adjacent chunk_index in a DIFFERENT section is a real chapter "
        "boundary, not a continuous passage, and must not be pulled in"
    )


def test_search_neighbor_expansion_does_not_duplicate_an_already_matched_chunk(
    tmp_path: Path,
) -> None:
    conn = open_store(tmp_path / "test.db")
    book = add_book(conn, "Book", "/inbox/b.pdf", "2026-01-01")
    add_chunk(conn, book, 0, "sec", "first", [1.0, 0.0])
    add_chunk(conn, book, 1, "sec", "second", [0.9, 0.1])
    commit(conn)

    # top_k=2 already includes both chunks directly -- neighbor expansion
    # of either one must not produce a duplicate entry for the other.
    results = search(conn, [1.0, 0.0], top_k=2)

    assert len(results) == 2
    assert sorted(r.text for r in results) == ["first", "second"]


def test_search_streams_via_fetchmany_not_fetchall() -> None:
    # BUG-7, 2026-07-30: search() used to fetchall() every chunk's embedding
    # before scoring -- fine at small scale, but a real 56,042-chunk library
    # made the raw embedding JSON alone ~852MB, OOM-killing library-keeper's
    # 512MB container on a real live query. sqlite3.Connection/Cursor are
    # immutable C types (verified: monkeypatching either raises
    # "attribute is read-only" at both the instance and class level), so
    # this can't assert the fix by counting real fetchmany() calls at
    # runtime -- it asserts the fix's actual mechanism via source inspection
    # instead: search()'s body must call fetchmany (bounded batches) and
    # must NOT call fetchall() on the main chunk-scan cursor. A regression
    # back to `.fetchall()` would pass every correctness test in this file
    # (see the sibling test below for those) while reintroducing the OOM --
    # this test exists specifically to catch that regression shape.
    source = inspect.getsource(search)
    assert "fetchmany" in source, "search() must page through chunks via cursor.fetchmany"
    assert ".fetchall()" not in source, (
        "search() must not fetchall() the chunk scan -- that's the exact "
        "regression (BUG-7) this test guards against"
    )


def test_search_correct_over_more_rows_than_one_batch(tmp_path: Path) -> None:
    # Companion to the source-inspection guard above: confirms the batched
    # scan still produces correct top-k results across a dataset larger
    # than a single internal batch, not just that batching syntax is present.
    conn = open_store(tmp_path / "test.db")
    book = add_book(conn, "Big Book", "/inbox/big.pdf", "2026-01-01")
    for i in range(1200):  # several times the batch size, one book is enough
        add_chunk(conn, book, i, "sec", f"chunk {i}", [1.0 if i == 0 else 0.0, 0.0])
    commit(conn)

    results = search(conn, [1.0, 0.0], top_k=3)

    assert results[0].text == "chunk 0"


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
            "INSERT INTO chunks (book_id, chunk_index, section, text, embedding) "
            "VALUES (1, ?, ?, ?, '[1.0]')",
            (i, f"page {i}", long_text),
        )
    old_conn.commit()
    old_conn.close()

    conn = open_store(db_path)

    statuses = list_book_statuses(conn)
    assert statuses[0].doc_type == "book"


def test_content_hash_backfill_retries_on_a_later_open_when_file_was_missing(
    tmp_path: Path,
) -> None:
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


def test_find_book_by_filename_matches_on_basename_regardless_of_prefix(tmp_path: Path) -> None:
    # docs/incidents/2026-07-30-sandbox-vs-ask-library.md: the caller's copy
    # of a path (e.g. a shell command argument) may use a different -- or
    # entirely fictitious -- directory prefix than library-parse's own
    # `/documents/<name>` mount. Only the basename can be relied on to match.
    conn = open_store(tmp_path / "test.db")
    add_book(
        conn,
        "Awakening the Heroes Within",
        "/documents/doc_8d70205d2619_Carol_S_Pearson_Awakening_the_Heroes_Within.epub",
        "2026-07-28",
        status="done",
    )

    match = find_book_by_filename(
        conn,
        "/opt/data/cache/documents/doc_8d70205d2619_Carol_S_Pearson_Awakening_the_Heroes_Within.epub",
    )

    assert match is not None
    assert match.title == "Awakening the Heroes Within"
    assert match.status == "done"


def test_find_book_by_filename_is_case_insensitive_and_accepts_bare_names(tmp_path: Path) -> None:
    conn = open_store(tmp_path / "test.db")
    add_book(conn, "Title", "/documents/Some_Book.PDF", "2026-01-01", status="done")

    assert find_book_by_filename(conn, "some_book.pdf") is not None
    assert find_book_by_filename(conn, "SOME_BOOK.PDF") is not None


def test_find_book_by_filename_no_match_returns_none(tmp_path: Path) -> None:
    conn = open_store(tmp_path / "test.db")
    add_book(conn, "Title", "/documents/a.pdf", "2026-01-01", status="done")

    assert find_book_by_filename(conn, "unrelated.pdf") is None
    assert find_book_by_filename(conn, "") is None


def test_find_book_by_filename_reflects_non_done_status(tmp_path: Path) -> None:
    # A match on a still-embedding book must NOT look interchangeable with a
    # done one -- the caller (sandbox_redirect plugin) only blocks/redirects
    # on status == "done", so the status has to travel through accurately.
    conn = open_store(tmp_path / "test.db")
    add_book(conn, "Title", "/documents/still_embedding.epub", "2026-01-01", status="embedding")

    match = find_book_by_filename(conn, "still_embedding.epub")

    assert match is not None
    assert match.status == "embedding"


def test_progress_and_status_tracking(tmp_path: Path) -> None:
    conn = open_store(tmp_path / "test.db")
    book_id = add_book(
        conn, "Title", "/inbox/a.pdf", "2026-01-01", status="embedding", total_chunks=10
    )

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
    record_knowledge_gap(
        conn, "what does the book say about X?", "no_matches", "2026-07-28T10:00:00"
    )

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
