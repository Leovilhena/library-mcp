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
    ingested_at TEXT NOT NULL
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


@dataclass(frozen=True, slots=True)
class SearchResult:
    book_title: str
    section: str | None
    text: str
    score: float


def open_store(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def add_book(conn: sqlite3.Connection, title: str, source_path: str, ingested_at: str) -> int:
    cur = conn.execute(
        "INSERT INTO books (title, source_path, ingested_at) VALUES (?, ?, ?)",
        (title, source_path, ingested_at),
    )
    conn.commit()
    return int(cur.lastrowid)  # type: ignore[arg-type]


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
