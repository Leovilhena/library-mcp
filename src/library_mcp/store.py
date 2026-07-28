"""Shared SQLite knowledge store: one database, every ingested book.

Deliberately one shared table across all books, not one database per book --
this is what makes cross-book retrieval ("networking book mentions Python,
surface it on a later Python question") fall out of plain similarity search
instead of needing an explicit link between documents.

No vector-DB engine and no numpy: at the scale of a personal library (tens
to low hundreds of thousands of chunks), a pure-Python cosine-similarity scan
is fast enough, and it keeps the image's dependency/CVE surface to what's
already needed for PDF/EPUB parsing.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS books (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    source_path TEXT NOT NULL,
    ingested_at TEXT NOT NULL,
    content_hash TEXT,
    status TEXT NOT NULL DEFAULT 'done',
    total_chunks INTEGER NOT NULL DEFAULT 0,
    embedded_chunks INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    book_id INTEGER NOT NULL REFERENCES books(id),
    chunk_index INTEGER NOT NULL,
    section TEXT,
    text TEXT NOT NULL,
    embedding TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_book ON chunks(book_id);
"""

# One partial unique index (content_hash IS NOT NULL) rather than a plain
# UNIQUE column -- some rows may legitimately have no hash (older code paths,
# or content that couldn't be hashed for some reason), and those should never
# collide with each other under a strict uniqueness constraint.
_CONTENT_HASH_INDEX = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_books_content_hash
ON books(content_hash) WHERE content_hash IS NOT NULL;
"""

# Columns added after the original schema shipped. `books` may already exist
# without them (this project's own real database did, on 2026-07-28) --
# CREATE TABLE IF NOT EXISTS does not retroactively add columns, so this
# migrates forward explicitly rather than assuming a fresh database.
_MIGRATION_COLUMNS = {
    "content_hash": "TEXT",
    "status": "TEXT NOT NULL DEFAULT 'done'",
    "total_chunks": "INTEGER NOT NULL DEFAULT 0",
    "embedded_chunks": "INTEGER NOT NULL DEFAULT 0",
}


class DuplicateBookError(Exception):
    """A book with this exact content is already in the store."""

    def __init__(self, title: str) -> None:
        super().__init__(f"already have this book: {title!r}")
        self.title = title


@dataclass(frozen=True, slots=True)
class SearchResult:
    book_title: str
    section: str | None
    text: str
    score: float


@dataclass(frozen=True, slots=True)
class BookStatus:
    title: str
    status: str
    total_chunks: int
    embedded_chunks: int


def _migrate(conn: sqlite3.Connection) -> None:
    existing = {row[1] for row in conn.execute("PRAGMA table_info(books)")}
    added_progress_columns = False
    added_hash_column = False
    for column, definition in _MIGRATION_COLUMNS.items():
        if column not in existing:
            conn.execute(f"ALTER TABLE books ADD COLUMN {column} {definition}")
            if column in ("total_chunks", "embedded_chunks"):
                added_progress_columns = True
            if column == "content_hash":
                added_hash_column = True
    if added_progress_columns:
        # New columns default to 0, but a book ingested before this feature
        # existed already has real chunks sitting in the chunks table --
        # without this, list_learned would show "(0 chunks)" for every
        # already-ingested book the moment this migration ran. Found live
        # 2026-07-28 against this project's own real database.
        conn.execute(
            """
            UPDATE books SET
                total_chunks = (SELECT COUNT(*) FROM chunks WHERE chunks.book_id = books.id),
                embedded_chunks = (SELECT COUNT(*) FROM chunks WHERE chunks.book_id = books.id)
            WHERE total_chunks = 0
            """
        )
    if added_hash_column:
        # A book ingested before dedup existed has no recorded hash, so a
        # re-upload of that exact file would sail past the duplicate check
        # with nothing to compare against -- found live 2026-07-28 re-adding
        # a book already in this project's own real database. Backfill by
        # re-reading source_path, when it's still a real file on disk (a
        # learn_text entry's source is a URL, not a file, and is correctly
        # left unhashed -- there's nothing to re-read for those).
        rows = conn.execute(
            "SELECT id, source_path FROM books WHERE content_hash IS NULL"
        ).fetchall()
        for book_id, source_path in rows:
            path = Path(source_path)
            if not path.is_file():
                continue
            content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            conn.execute(
                "UPDATE books SET content_hash = ? WHERE id = ?", (content_hash, book_id)
            )
    conn.commit()


def open_store(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.executescript(_SCHEMA)
    conn.commit()
    _migrate(conn)
    # Separate from _SCHEMA: needs the column to exist first, which on an
    # old database only just happened via _migrate above.
    conn.execute(_CONTENT_HASH_INDEX)
    conn.commit()
    return conn


def find_book_by_hash(conn: sqlite3.Connection, content_hash: str) -> str | None:
    row = conn.execute(
        "SELECT title FROM books WHERE content_hash = ?", (content_hash,)
    ).fetchone()
    return row[0] if row else None


def add_book(
    conn: sqlite3.Connection,
    title: str,
    source_path: str,
    ingested_at: str,
    content_hash: str | None = None,
    status: str = "done",
    total_chunks: int = 0,
) -> int:
    try:
        cur = conn.execute(
            "INSERT INTO books (title, source_path, ingested_at, content_hash, status, total_chunks) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (title, source_path, ingested_at, content_hash, status, total_chunks),
        )
    except sqlite3.IntegrityError as exc:
        # A race: two concurrent learn() calls for the same content. The
        # earlier, slower check-then-insert isn't atomic across that gap;
        # the unique index is what actually prevents the duplicate row.
        existing = find_book_by_hash(conn, content_hash) if content_hash else None
        raise DuplicateBookError(existing or title) from exc
    conn.commit()
    return int(cur.lastrowid)  # type: ignore[arg-type]


def update_book_progress(conn: sqlite3.Connection, book_id: int, embedded_chunks: int) -> None:
    conn.execute("UPDATE books SET embedded_chunks = ? WHERE id = ?", (embedded_chunks, book_id))
    conn.commit()


def mark_book_status(conn: sqlite3.Connection, book_id: int, status: str) -> None:
    conn.execute("UPDATE books SET status = ? WHERE id = ?", (status, book_id))
    conn.commit()


def delete_book(conn: sqlite3.Connection, book_id: int) -> None:
    conn.execute("DELETE FROM chunks WHERE book_id = ?", (book_id,))
    conn.execute("DELETE FROM books WHERE id = ?", (book_id,))
    conn.commit()


def add_chunk(
    conn: sqlite3.Connection,
    book_id: int,
    chunk_index: int,
    section: str | None,
    text: str,
    embedding: list[float],
) -> None:
    conn.execute(
        "INSERT INTO chunks (book_id, chunk_index, section, text, embedding) "
        "VALUES (?, ?, ?, ?, ?)",
        (book_id, chunk_index, section, text, json.dumps(embedding)),
    )


def commit(conn: sqlite3.Connection) -> None:
    conn.commit()


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def search(conn: sqlite3.Connection, query_embedding: list[float], top_k: int) -> list[SearchResult]:
    rows = conn.execute(
        "SELECT chunks.section, chunks.text, chunks.embedding, books.title "
        "FROM chunks JOIN books ON chunks.book_id = books.id"
    ).fetchall()
    scored: list[SearchResult] = []
    for section, text, embedding_json, title in rows:
        embedding = json.loads(embedding_json)
        score = _cosine(query_embedding, embedding)
        scored.append(SearchResult(book_title=title, section=section, text=text, score=score))
    scored.sort(key=lambda r: r.score, reverse=True)
    return scored[:top_k]


def book_count(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(*) FROM books").fetchone()
    return int(row[0])


def chunk_count(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
    return int(row[0])


def list_books(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute("SELECT title FROM books ORDER BY id").fetchall()
    return [r[0] for r in rows]


def list_book_statuses(conn: sqlite3.Connection) -> list[BookStatus]:
    rows = conn.execute(
        "SELECT title, status, total_chunks, embedded_chunks FROM books ORDER BY id"
    ).fetchall()
    return [BookStatus(title=t, status=s, total_chunks=tc, embedded_chunks=ec) for t, s, tc, ec in rows]
