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
import heapq
import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from library_mcp.parser import TextBlock, detect_doc_type

_SCHEMA = """
CREATE TABLE IF NOT EXISTS books (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    source_path TEXT NOT NULL,
    ingested_at TEXT NOT NULL,
    content_hash TEXT,
    status TEXT NOT NULL DEFAULT 'done',
    total_chunks INTEGER NOT NULL DEFAULT 0,
    embedded_chunks INTEGER NOT NULL DEFAULT 0,
    doc_type TEXT
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
CREATE TABLE IF NOT EXISTS knowledge_gaps (
    id INTEGER PRIMARY KEY,
    question TEXT NOT NULL,
    reason TEXT NOT NULL,
    first_asked_at TEXT NOT NULL,
    last_asked_at TEXT NOT NULL,
    times_asked INTEGER NOT NULL DEFAULT 1,
    resolved INTEGER NOT NULL DEFAULT 0
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_knowledge_gaps_question
ON knowledge_gaps(question) WHERE resolved = 0;

-- Knowledge-gap research (docs/planning/knowledge-gap-research.md §4/§9
-- step 0). Same database, new tables -- no sibling file: no hot-path
-- mtime-watcher reads knowledge.db the way tgproxy reads reflex.db, so the
-- reason that split exists doesn't transfer here.
CREATE TABLE IF NOT EXISTS external_research (
    id INTEGER PRIMARY KEY,
    gap_id INTEGER NOT NULL REFERENCES knowledge_gaps(id),
    question TEXT NOT NULL,
    source TEXT NOT NULL,
    source_title TEXT,
    source_url TEXT,
    extract TEXT,
    synthesized_answer TEXT,
    outcome TEXT NOT NULL,
    researched_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_external_research_gap ON external_research(gap_id);

CREATE TABLE IF NOT EXISTS pending_followups (
    id INTEGER PRIMARY KEY,
    gap_id INTEGER NOT NULL REFERENCES knowledge_gaps(id),
    chat_id TEXT NOT NULL,
    question TEXT NOT NULL,
    resolution_kind TEXT NOT NULL,
    summary TEXT NOT NULL,
    resolved_at TEXT NOT NULL,
    injected_at TEXT,
    expired_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_pending_followups_open
ON pending_followups(injected_at, expired_at) WHERE injected_at IS NULL AND expired_at IS NULL;

-- Book structuring (docs/planning/book-structuring.md §6/§9 step 0). Same
-- database, same additive pattern -- no sibling file, same reasoning as
-- knowledge-gap research's own tables above.
CREATE TABLE IF NOT EXISTS book_chapters (
    id INTEGER PRIMARY KEY,
    book_id INTEGER NOT NULL REFERENCES books(id),
    chapter_index INTEGER NOT NULL,
    section TEXT NOT NULL,
    summary TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    model TEXT,
    structured_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_book_chapters_book ON book_chapters(book_id);

CREATE TABLE IF NOT EXISTS book_glossary_terms (
    id INTEGER PRIMARY KEY,
    book_id INTEGER NOT NULL REFERENCES books(id),
    chapter_id INTEGER REFERENCES book_chapters(id),
    term TEXT NOT NULL,
    definition TEXT NOT NULL,
    model TEXT,
    extracted_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_book_glossary_book ON book_glossary_terms(book_id);
CREATE INDEX IF NOT EXISTS idx_book_glossary_term ON book_glossary_terms(term);
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
    "doc_type": "TEXT",
}

# Knowledge-gap research (docs/planning/knowledge-gap-research.md §4/§7):
# two additive columns on the pre-existing `knowledge_gaps` table, same
# migration pattern as `_MIGRATION_COLUMNS` above, applied to a different
# table so kept separate rather than merged into one dict keyed by table.
_KNOWLEDGE_GAPS_MIGRATION_COLUMNS = {
    "external_attempt_count": "INTEGER NOT NULL DEFAULT 0",
    "giving_up": "INTEGER NOT NULL DEFAULT 0",
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
class KnowledgeGap:
    question: str
    reason: str
    first_asked_at: str
    last_asked_at: str
    times_asked: int


@dataclass(frozen=True, slots=True)
class BookStatus:
    title: str
    status: str
    total_chunks: int
    embedded_chunks: int
    doc_type: str | None


@dataclass(frozen=True, slots=True)
class OpenGap:
    """One `knowledge_gaps` row, id included -- unlike `KnowledgeGap` above,
    which is Leo's own read-only `list_knowledge_gaps` view and deliberately
    doesn't expose the internal id. The nightly gap-research job (§9 step 6)
    needs the id to write `external_research`/`pending_followups` rows
    against it."""

    id: int
    question: str
    reason: str
    times_asked: int
    external_attempt_count: int
    giving_up: bool


@dataclass(frozen=True, slots=True)
class ExternalResearchRow:
    id: int
    gap_id: int
    source: str
    outcome: str
    researched_at: str


@dataclass(frozen=True, slots=True)
class PendingFollowup:
    id: int
    gap_id: int
    chat_id: str
    question: str
    resolution_kind: str
    summary: str
    resolved_at: str


@dataclass(frozen=True, slots=True)
class BookChapter:
    """One `book_chapters` row (docs/planning/book-structuring.md §6)."""

    id: int
    book_id: int
    chapter_index: int
    section: str
    status: str
    model: str | None
    structured_at: str | None


@dataclass(frozen=True, slots=True)
class GlossaryEntry:
    """One glossary hit, joined with its book title -- what `_ask()`'s
    term-lookup pre-check (§4, §9 step 4) actually needs, not a raw row."""

    term: str
    definition: str
    book_title: str
    chapter_index: int | None
    model: str | None


@dataclass(frozen=True, slots=True)
class StructuringCandidate:
    """One EPUB book selected for a `book_structure.py` run (§9 step 3)."""

    book_id: int
    title: str
    source_path: str
    ingested_at: str


@dataclass(frozen=True, slots=True)
class BookStructuringStatus:
    """§7's one small addition: a `structured: yes/no/pending` flag per book
    for an existing `list_learned`-style view, not a new tool."""

    title: str
    structured: str  # "yes" | "pending" | "no"


def _migrate(conn: sqlite3.Connection) -> None:
    existing = {row[1] for row in conn.execute("PRAGMA table_info(books)")}
    for column, definition in _MIGRATION_COLUMNS.items():
        if column not in existing:
            conn.execute(f"ALTER TABLE books ADD COLUMN {column} {definition}")

    existing_gap_columns = {row[1] for row in conn.execute("PRAGMA table_info(knowledge_gaps)")}
    for column, definition in _KNOWLEDGE_GAPS_MIGRATION_COLUMNS.items():
        if column not in existing_gap_columns:
            conn.execute(f"ALTER TABLE knowledge_gaps ADD COLUMN {column} {definition}")

    # All three backfills below run unconditionally on every open_store()
    # call, not gated on "did *this* ALTER TABLE just run" -- deliberately.
    # library-parse and library-keeper are two separate processes sharing
    # this one database with no `depends_on` ordering between them, and only
    # library-parse's container mounts /inbox and /documents. A real,
    # previously-live bug: if library-keeper happened to be the process that
    # added `content_hash` (e.g. it opened the db first after a fresh
    # migration), the backfill ran there instead, found zero readable
    # source files (keeper has no filesystem mount for them), and the
    # gate variable being process-local meant it never got a second try
    # when library-parse (which *can* read the files) opened the db later.
    # Each backfill's own WHERE clause already makes it a safe no-op when
    # there's nothing left to do, so unconditional is strictly safer than
    # "once, in whichever process happened to add the column."

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

    # A book ingested before dedup existed has no recorded hash, so a
    # re-upload of that exact file would sail past the duplicate check
    # with nothing to compare against -- found live 2026-07-28 re-adding
    # a book already in this project's own real database. Backfill by
    # re-reading source_path, when it's still a real file on disk (a
    # learn_text entry's source is a URL, not a file, and is correctly
    # left unhashed -- there's nothing to re-read for those).
    rows = conn.execute("SELECT id, source_path FROM books WHERE content_hash IS NULL").fetchall()
    for book_id, source_path in rows:
        path = Path(source_path)
        if not path.is_file():
            continue
        content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        conn.execute("UPDATE books SET content_hash = ? WHERE id = ?", (content_hash, book_id))
    # Classify books ingested before this feature existed, same "don't
    # leave already-ingested data unclassified" requirement as the
    # content_hash backfill above. Reuses chunks already stored in the
    # `chunks` table rather than re-reading source files -- those may
    # not exist any more (learn_text sources are URLs, not files) or may
    # have moved, and the chunk text is exactly what the classifier
    # needs anyway.
    rows = conn.execute("SELECT id, source_path FROM books WHERE doc_type IS NULL").fetchall()
    for book_id, source_path in rows:
        chunk_rows = conn.execute(
            "SELECT section, text FROM chunks WHERE book_id = ? ORDER BY chunk_index",
            (book_id,),
        ).fetchall()
        blocks = [TextBlock(section=s, text=t) for s, t in chunk_rows]
        suffix = Path(source_path).suffix.lower() or None
        doc_type = detect_doc_type(blocks, suffix)
        conn.execute("UPDATE books SET doc_type = ? WHERE id = ?", (doc_type, book_id))
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
    row = conn.execute("SELECT title FROM books WHERE content_hash = ?", (content_hash,)).fetchone()
    return row[0] if row else None


@dataclass(frozen=True, slots=True)
class BookLookup:
    """One book match for `find_book_by_filename` -- title/source_path/status,
    everything the sandbox-redirect plugin (docs/incidents/2026-07-30
    -sandbox-vs-ask-library.md) needs to decide whether a shell command's
    file argument already has a done, fully-searchable `ask_library` path."""

    title: str
    source_path: str
    status: str


def find_book_by_filename(conn: sqlite3.Connection, filename: str) -> BookLookup | None:
    """Match a bare filename or a path fragment against `books.source_path`
    by basename, regardless of the directory prefix the caller's copy of
    the path uses -- library-parse stores paths as `/documents/<name>` in
    its own container's mount namespace, but a shell command elsewhere (e.g.
    sandbox-exec, which has no mount into the library at all) may reference
    the same file under a different or entirely fictitious prefix. Basename
    matching is the only thing both sides can agree on.

    `filename` may itself be a full path or just a bare name -- only its
    own basename is used for the comparison. Case-insensitive: filenames
    with mixed-case originals (e.g. from a Telegram upload) shouldn't need
    an exact-case shell argument to match. Returns the first match if
    duplicates somehow exist (source_path is not unique in the schema)."""
    needle = Path(filename).name.strip().lower()
    if not needle:
        return None
    rows = conn.execute("SELECT title, source_path, status FROM books").fetchall()
    for title, source_path, status in rows:
        if Path(source_path).name.strip().lower() == needle:
            return BookLookup(title=title, source_path=source_path, status=status)
    return None


def add_book(
    conn: sqlite3.Connection,
    title: str,
    source_path: str,
    ingested_at: str,
    content_hash: str | None = None,
    status: str = "done",
    total_chunks: int = 0,
    doc_type: str | None = None,
) -> int:
    try:
        cur = conn.execute(
            "INSERT INTO books "
            "(title, source_path, ingested_at, content_hash, status, total_chunks, doc_type) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (title, source_path, ingested_at, content_hash, status, total_chunks, doc_type),
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


def find_books_by_title_substring(conn: sqlite3.Connection, query: str) -> list[tuple[int, str]]:
    """Case-insensitive substring match against book titles.

    Returns every match rather than guessing the best one -- an ambiguous
    query (multiple books matching) should be refused back to the caller
    to disambiguate, same "don't invent certainty" principle already used
    for fuzzy filename matching in parse_server.py.
    """
    # Escape the escape character itself FIRST -- found by review: escaping
    # '%'/'_' without first escaping '\' means a literal backslash in the
    # query (e.g. a Windows-style path fragment) gets silently consumed as
    # part of whatever escape sequence follows it, and the intended
    # substring never matches.
    escaped = query.replace("\\", "\\\\").replace("%", r"\%").replace("_", r"\_")
    rows = conn.execute(
        "SELECT id, title FROM books WHERE title LIKE ? ESCAPE '\\' ORDER BY id",
        (f"%{escaped}%",),
    ).fetchall()
    return [(int(i), t) for i, t in rows]


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


def record_knowledge_gap(
    conn: sqlite3.Connection, question: str, reason: str, asked_at: str
) -> int:
    """Log a question the keeper couldn't answer -- deterministic call sites only.

    Called from two structural, code-driven signals in keeper_server.py's
    `_ask()` -- an AnswerDecision reached with no search results backing it
    (the actually-reachable "we have nothing" case: the keeper's own prompt
    tells the reasoner to answer regardless on its last attempt, so this
    surfaces as a real answer, not a loop exhausting silently), or the
    reasoner repeating the same search query twice in a row -- never from
    parsing the model's own answer text for phrases like "I don't know",
    which would depend on wording this stack's own model is only 40-60%
    reliable at producing consistently.

    Re-asking the same exact question bumps `times_asked` and refreshes
    `last_asked_at` rather than creating a duplicate row, so a question asked
    by three different people shows up once, with a real repeat count.

    Returns the row's `id` (docs/planning/book-qa-escalation-flow.md §2/§4.3):
    `keeper_server.py`'s `AskOutcome.gap_id` needs the id of the row just
    written or bumped this call, without a second query to re-derive it.
    """
    existing = conn.execute(
        "SELECT id, times_asked FROM knowledge_gaps WHERE question = ? AND resolved = 0",
        (question,),
    ).fetchone()
    if existing is not None:
        gap_id, times_asked = existing
        conn.execute(
            "UPDATE knowledge_gaps SET last_asked_at = ?, times_asked = ? WHERE id = ?",
            (asked_at, times_asked + 1, gap_id),
        )
    else:
        cursor = conn.execute(
            "INSERT INTO knowledge_gaps "
            "(question, reason, first_asked_at, last_asked_at, times_asked) "
            "VALUES (?, ?, ?, ?, 1)",
            (question, reason, asked_at, asked_at),
        )
        gap_id = cursor.lastrowid
        # lastrowid is only None when no INSERT has happened on this cursor
        # yet, which can't be true here -- the INSERT just above is what
        # this is reading it from. A real check, not just satisfying mypy:
        # `assert` gets stripped under `python -O`, so this can't rely on
        # one for something that would otherwise silently return a bogus
        # id to a caller (bandit B101 -- correctly flagged, not suppressed).
        if gap_id is None:
            msg = "INSERT into knowledge_gaps did not produce a lastrowid"
            raise RuntimeError(msg)
    conn.commit()
    return int(gap_id)


def list_knowledge_gaps(conn: sqlite3.Connection) -> list[KnowledgeGap]:
    rows = conn.execute(
        "SELECT question, reason, first_asked_at, last_asked_at, times_asked "
        "FROM knowledge_gaps WHERE resolved = 0 "
        "ORDER BY times_asked DESC, last_asked_at DESC"
    ).fetchall()
    return [
        KnowledgeGap(question=q, reason=r, first_asked_at=fa, last_asked_at=la, times_asked=ta)
        for q, r, fa, la, ta in rows
    ]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# Module-level (not a local inside search()) so it reads as the real tuning
# constant it is, rather than a per-call magic number.
_SEARCH_BATCH_SIZE = 500


def search(
    conn: sqlite3.Connection, query_embedding: list[float], top_k: int
) -> list[SearchResult]:
    """Top-k semantic search, plus each match's immediate same-section
    neighbors (docs/TODO.md BUG-1, root-caused 2026-07-29).

    Books are chunked by fixed size, not content boundary, so a single
    continuous passage (e.g. a ~50-item list spanning ~3 pages) can land
    across several consecutive `chunk_index` rows in one `section`. A bare
    top-k semantic match doesn't score all of them equally against a
    generic query ("list the values"), so a real answer came back with 2 of
    ~50 items -- confirmed directly against a live book (`chunk_index`
    221-225, one section, only 1-2 chunks surfacing per query). Pulling in
    `chunk_index - 1`/`chunk_index + 1` from the same book *and* the same
    section (never crossing a section boundary -- an adjacent index in a
    different chapter is not the same continuous passage) closes that gap
    without needing to guess a better query or raise `top_k` blindly.

    Neighbors are ADDITIVE, not counted against `top_k` -- the top-k
    ranking itself is unchanged, this only fills in the surrounding context
    once a passage has already been judged relevant. The result list can
    therefore be longer than `top_k`; callers already bound total context
    by character budget (`_format_context`'s `max_context_chars`), not by
    list length, so this doesn't reopen the prompt-budget problem raised
    elsewhere in this stack's docs.

    Real incident, 2026-07-30 (BUG-7): this used to `fetchall()` every
    chunk's embedding into Python memory before scoring, then sort the
    whole thing. Fine at small scale; at 56,042 real chunks the raw
    embedding JSON text alone was ~852MB, well past `library-keeper`'s
    memory limit -- OOM-killed three times in a row on a real live query,
    confirmed via `dmesg`. Fixed by streaming the cursor in bounded
    batches and keeping only a `top_k`-sized min-heap of scores in memory
    at any point, rather than materializing every row at once. Peak memory
    is now O(batch_size + top_k), not O(total_chunks) -- this scales with
    library size again instead of degrading every time a new book is
    learned.
    """
    cursor = conn.execute(
        "SELECT chunks.id, chunks.book_id, chunks.chunk_index, chunks.section, "
        "chunks.text, chunks.embedding, books.title "
        "FROM chunks JOIN books ON chunks.book_id = books.id"
    )
    # A plain min-heap keyed on score alone would break the moment two rows
    # tie exactly (heapq falls through to comparing the rest of the tuple,
    # and `str`/`None` section values aren't always orderable against each
    # other) -- the monotonically increasing `counter` is a cheap, always-
    # comparable tiebreaker that keeps heap ordering well-defined without
    # ever touching row content for comparison.
    heap: list[tuple[float, int, tuple[int, int, int, str | None, str, str]]] = []
    counter = 0
    while True:
        batch = cursor.fetchmany(_SEARCH_BATCH_SIZE)
        if not batch:
            break
        for chunk_id, book_id, chunk_index, section, text, embedding_json, title in batch:
            embedding = json.loads(embedding_json)
            score = _cosine(query_embedding, embedding)
            row_data = (chunk_id, book_id, chunk_index, section, text, title)
            if len(heap) < top_k:
                heapq.heappush(heap, (score, counter, row_data))
            elif top_k > 0 and score > heap[0][0]:
                heapq.heapreplace(heap, (score, counter, row_data))
            counter += 1
    top = [(score, *row_data) for score, _counter, row_data in sorted(heap, reverse=True)]

    seen_chunk_ids = {chunk_id for _, chunk_id, *_ in top}
    results = [
        SearchResult(book_title=title, section=section, text=text, score=score)
        for score, _chunk_id, _book_id, _chunk_index, section, text, title in top
    ]

    for score, _chunk_id, book_id, chunk_index, section, _text, title in top:
        for neighbor_index in (chunk_index - 1, chunk_index + 1):
            neighbor = conn.execute(
                "SELECT id, section, text FROM chunks WHERE book_id = ? AND chunk_index = ?",
                (book_id, neighbor_index),
            ).fetchone()
            if neighbor is None:
                continue
            neighbor_id, neighbor_section, neighbor_text = neighbor
            if neighbor_id in seen_chunk_ids:
                continue
            if neighbor_section != section:
                # Adjacent chunk_index but a different section -- the fixed-
                # size chunker crossed a real content boundary here, so this
                # is not part of the same continuous passage.
                continue
            seen_chunk_ids.add(neighbor_id)
            results.append(
                SearchResult(
                    book_title=title, section=neighbor_section, text=neighbor_text, score=score
                )
            )
    return results


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
        "SELECT title, status, total_chunks, embedded_chunks, doc_type FROM books ORDER BY id"
    ).fetchall()
    return [
        BookStatus(title=t, status=s, total_chunks=tc, embedded_chunks=ec, doc_type=dt)
        for t, s, tc, ec, dt in rows
    ]


# ---------------------------------------------------------------------------
# Knowledge-gap research (docs/planning/knowledge-gap-research.md)
# ---------------------------------------------------------------------------


def list_open_gaps(conn: sqlite3.Connection) -> list[OpenGap]:
    """Every unresolved gap, id included -- the nightly job's triage input
    (§9 step 6). Ordered oldest-asked-first so a gap sitting open the
    longest gets researched first when a run has to stop partway."""
    rows = conn.execute(
        "SELECT id, question, reason, times_asked, external_attempt_count, giving_up "
        "FROM knowledge_gaps WHERE resolved = 0 ORDER BY first_asked_at"
    ).fetchall()
    return [
        OpenGap(
            id=int(i),
            question=q,
            reason=r,
            times_asked=int(ta),
            external_attempt_count=int(eac),
            giving_up=bool(gu),
        )
        for i, q, r, ta, eac, gu in rows
    ]


def mark_gap_resolved(conn: sqlite3.Connection, gap_id: int) -> None:
    conn.execute("UPDATE knowledge_gaps SET resolved = 1 WHERE id = ?", (gap_id,))
    conn.commit()


def increment_external_attempt_count(conn: sqlite3.Connection, gap_id: int) -> int:
    """Bump the retry-cap counter and return the new value.

    Only called on a `content_miss` outcome (§3.3/§7) -- `infra_down`
    attempts never reach here, which is the whole point of the split: an
    outage says nothing about whether the content exists and must never
    spend down the same budget a genuine "no" spends.
    """
    conn.execute(
        "UPDATE knowledge_gaps SET external_attempt_count = external_attempt_count + 1 "
        "WHERE id = ?",
        (gap_id,),
    )
    conn.commit()
    row = conn.execute(
        "SELECT external_attempt_count FROM knowledge_gaps WHERE id = ?", (gap_id,)
    ).fetchone()
    return int(row[0]) if row else 0


def set_giving_up(conn: sqlite3.Connection, gap_id: int) -> None:
    """The hard retry cap (§7.1) tripped: external research stops
    permanently for this gap. Book-deepen keeps retrying regardless --
    `giving_up` only gates the external-research half of the pipeline."""
    conn.execute("UPDATE knowledge_gaps SET giving_up = 1 WHERE id = ?", (gap_id,))
    conn.commit()


def record_external_research(
    conn: sqlite3.Connection,
    gap_id: int,
    question: str,
    source: str,
    outcome: str,
    researched_at: str,
    source_title: str | None = None,
    source_url: str | None = None,
    extract: str | None = None,
    synthesized_answer: str | None = None,
) -> int:
    """Append one external-source attempt, whatever its outcome (§4/§7) --
    even an `infra_down` attempt is written, so the attempt itself is on
    record and auditable, it just doesn't count toward the retry cap."""
    cur = conn.execute(
        "INSERT INTO external_research "
        "(gap_id, question, source, source_title, source_url, extract, "
        "synthesized_answer, outcome, researched_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            gap_id,
            question,
            source,
            source_title,
            source_url,
            extract,
            synthesized_answer,
            outcome,
            researched_at,
        ),
    )
    conn.commit()
    return int(cur.lastrowid)  # type: ignore[arg-type]


def latest_external_research(
    conn: sqlite3.Connection, gap_id: int, source: str
) -> ExternalResearchRow | None:
    """Most recent attempt for this gap/source pair -- backs the 7-day
    content-miss cooldown (§7.1): the caller checks `outcome`/`researched_at`
    on the result to decide whether to skip this run."""
    row = conn.execute(
        "SELECT id, gap_id, source, outcome, researched_at FROM external_research "
        "WHERE gap_id = ? AND source = ? ORDER BY researched_at DESC LIMIT 1",
        (gap_id, source),
    ).fetchone()
    if row is None:
        return None
    i, g, s, o, ra = row
    return ExternalResearchRow(id=int(i), gap_id=int(g), source=s, outcome=o, researched_at=ra)


def create_pending_followup(
    conn: sqlite3.Connection,
    gap_id: int,
    chat_id: str,
    question: str,
    resolution_kind: str,
    summary: str,
    resolved_at: str,
) -> int:
    cur = conn.execute(
        "INSERT INTO pending_followups "
        "(gap_id, chat_id, question, resolution_kind, summary, resolved_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (gap_id, chat_id, question, resolution_kind, summary, resolved_at),
    )
    conn.commit()
    return int(cur.lastrowid)  # type: ignore[arg-type]


def list_pending_followups(
    conn: sqlite3.Connection, chat_id: str, limit: int = 2
) -> list[PendingFollowup]:
    """Resolved-but-not-yet-surfaced followups for one chat (§5.1) --
    plain, deterministic SELECT, the only thing that decides what gets
    handed to a live turn. `chat_id` isn't really filtered on in today's
    single-user deployment in spirit, but the column and the WHERE clause
    are real now so adding real multi-chat filtering later is additive."""
    rows = conn.execute(
        "SELECT id, gap_id, chat_id, question, resolution_kind, summary, resolved_at "
        "FROM pending_followups "
        "WHERE chat_id = ? AND injected_at IS NULL AND expired_at IS NULL "
        "ORDER BY resolved_at LIMIT ?",
        (chat_id, limit),
    ).fetchall()
    return [
        PendingFollowup(
            id=int(i),
            gap_id=int(g),
            chat_id=c,
            question=q,
            resolution_kind=rk,
            summary=s,
            resolved_at=ra,
        )
        for i, g, c, q, rk, s, ra in rows
    ]


def mark_followups_delivered(
    conn: sqlite3.Connection, followup_ids: list[int], injected_at: str
) -> list[int]:
    """Set `injected_at` for each id that's still pending. Idempotent: an
    already-delivered or already-expired id is left untouched and simply
    isn't returned, matching the MCP tool's documented no-op-on-repeat
    contract (§5.1)."""
    delivered: list[int] = []
    for followup_id in followup_ids:
        cur = conn.execute(
            "UPDATE pending_followups SET injected_at = ? "
            "WHERE id = ? AND injected_at IS NULL AND expired_at IS NULL",
            (injected_at, followup_id),
        )
        if cur.rowcount:
            delivered.append(followup_id)
    conn.commit()
    return delivered


def expire_stale_followups(
    conn: sqlite3.Connection, now: str, cutoff: str
) -> list[PendingFollowup]:
    """21-day staleness sweep (§5.3): any never-surfaced followup resolved
    before `cutoff` gets `expired_at` set. Returns the rows expired so the
    caller can write one `followup_expired` audit event per row. Expiry only
    affects the proactive-aside channel -- the underlying `knowledge_gaps`
    row stays resolved and `external_research` stays on record."""
    rows = conn.execute(
        "SELECT id, gap_id, chat_id, question, resolution_kind, summary, resolved_at "
        "FROM pending_followups "
        "WHERE injected_at IS NULL AND expired_at IS NULL AND resolved_at < ?",
        (cutoff,),
    ).fetchall()
    expired = [
        PendingFollowup(
            id=int(i),
            gap_id=int(g),
            chat_id=c,
            question=q,
            resolution_kind=rk,
            summary=s,
            resolved_at=ra,
        )
        for i, g, c, q, rk, s, ra in rows
    ]
    if expired:
        conn.executemany(
            "UPDATE pending_followups SET expired_at = ? WHERE id = ?",
            [(now, f.id) for f in expired],
        )
        conn.commit()
    return expired


# ---------------------------------------------------------------------------
# Book structuring (docs/planning/book-structuring.md §6/§9)
# ---------------------------------------------------------------------------


def list_epub_books_needing_structuring(
    conn: sqlite3.Connection, limit: int
) -> list[StructuringCandidate]:
    """Oldest-unstructured-first, EPUB only (§2, §8, §9 step 3).

    "Unstructured" covers two cases, both real and both worth re-selecting:
    a book with no `book_chapters` rows at all (never attempted), and a book
    that has rows but at least one is still `status='pending'` (a previous
    run started it and stopped -- crashed, hit the batch limit mid-book,
    whatever -- before finishing every chapter). `book_chapters.status`
    exists specifically so partial progress is visible and resumable (§6's
    own comment on the column); this query is what makes that real rather
    than aspirational. A book whose chapters are all `done`/`failed` is
    fully processed and never resurfaces here.
    """
    rows = conn.execute(
        "SELECT id, title, source_path, ingested_at FROM books "
        "WHERE source_path LIKE '%.epub' "
        "AND ("
        "  id NOT IN (SELECT DISTINCT book_id FROM book_chapters)"
        "  OR id IN (SELECT DISTINCT book_id FROM book_chapters WHERE status = 'pending')"
        ") "
        "ORDER BY ingested_at LIMIT ?",
        (limit,),
    ).fetchall()
    return [
        StructuringCandidate(book_id=int(i), title=t, source_path=sp, ingested_at=ia)
        for i, t, sp, ia in rows
    ]


def list_sections_for_book(conn: sqlite3.Connection, book_id: int) -> list[tuple[str, str]]:
    """Reconstruct this book's chapter-shaped blocks straight from `chunks`,
    without touching the source file again (§9 step 1: "no new parsing
    code -- the boundary is already implicit in `section`"). Groups chunks
    by `section`, preserving first-appearance order (== chapter order, since
    `extract_epub` emits one section per spine item in document order and
    `chunk_blocks` never reorders), and concatenates each section's chunk
    texts in `chunk_index` order. `library-keeper` has no filesystem mount
    for the original EPUB -- only `library-parse` does -- so this is also
    the only source of chapter text this process can reach.
    """
    rows = conn.execute(
        "SELECT section, text FROM chunks WHERE book_id = ? AND section IS NOT NULL "
        "ORDER BY chunk_index",
        (book_id,),
    ).fetchall()
    order: list[str] = []
    parts: dict[str, list[str]] = {}
    for section, text in rows:
        if section not in parts:
            parts[section] = []
            order.append(section)
        parts[section].append(text)
    return [(section, "\n".join(parts[section])) for section in order]


def add_chapter(conn: sqlite3.Connection, book_id: int, chapter_index: int, section: str) -> int:
    cur = conn.execute(
        "INSERT INTO book_chapters (book_id, chapter_index, section, status) "
        "VALUES (?, ?, ?, 'pending')",
        (book_id, chapter_index, section),
    )
    conn.commit()
    return int(cur.lastrowid)  # type: ignore[arg-type]


def mark_chapter_status(
    conn: sqlite3.Connection,
    chapter_id: int,
    status: str,
    model: str | None = None,
    structured_at: str | None = None,
) -> None:
    conn.execute(
        "UPDATE book_chapters SET status = ?, "
        "model = COALESCE(?, model), "
        "structured_at = COALESCE(?, structured_at) "
        "WHERE id = ?",
        (status, model, structured_at, chapter_id),
    )
    conn.commit()


def list_chapters_for_book(conn: sqlite3.Connection, book_id: int) -> list[BookChapter]:
    rows = conn.execute(
        "SELECT id, book_id, chapter_index, section, status, model, structured_at "
        "FROM book_chapters WHERE book_id = ? ORDER BY chapter_index",
        (book_id,),
    ).fetchall()
    return [
        BookChapter(
            id=int(i),
            book_id=int(b),
            chapter_index=int(ci),
            section=s,
            status=status,
            model=m,
            structured_at=sa,
        )
        for i, b, ci, s, status, m, sa in rows
    ]


def add_glossary_term(
    conn: sqlite3.Connection,
    book_id: int,
    chapter_id: int,
    term: str,
    definition: str,
    model: str,
    extracted_at: str,
) -> int:
    cur = conn.execute(
        "INSERT INTO book_glossary_terms "
        "(book_id, chapter_id, term, definition, model, extracted_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (book_id, chapter_id, term, definition, model, extracted_at),
    )
    conn.commit()
    return int(cur.lastrowid)  # type: ignore[arg-type]


def lookup_glossary_term(conn: sqlite3.Connection, term: str) -> GlossaryEntry | None:
    """Deterministic exact-then-substring match against `term` (§6, §9 step
    4) -- no embedding search, no small-model judgment, same house-rule fit
    as `find_books_by_title_substring`'s own `LIKE`-based matching. An
    exact case-insensitive match wins over a substring match so a precise
    term ("recursion") isn't shadowed by a longer stored term that happens
    to contain it ("recursion theorem") when both exist.
    """
    row = conn.execute(
        "SELECT book_glossary_terms.term, book_glossary_terms.definition, "
        "books.title, book_chapters.chapter_index, book_glossary_terms.model "
        "FROM book_glossary_terms "
        "JOIN books ON book_glossary_terms.book_id = books.id "
        "LEFT JOIN book_chapters ON book_glossary_terms.chapter_id = book_chapters.id "
        "WHERE lower(book_glossary_terms.term) = lower(?) "
        "ORDER BY book_glossary_terms.extracted_at DESC LIMIT 1",
        (term,),
    ).fetchone()
    if row is None:
        # Same escape-the-escape-character-first reasoning as
        # find_books_by_title_substring.
        escaped = term.replace("\\", "\\\\").replace("%", r"\%").replace("_", r"\_")
        row = conn.execute(
            "SELECT book_glossary_terms.term, book_glossary_terms.definition, "
            "books.title, book_chapters.chapter_index, book_glossary_terms.model "
            "FROM book_glossary_terms "
            "JOIN books ON book_glossary_terms.book_id = books.id "
            "LEFT JOIN book_chapters ON book_glossary_terms.chapter_id = book_chapters.id "
            "WHERE book_glossary_terms.term LIKE ? ESCAPE '\\' "
            "ORDER BY book_glossary_terms.extracted_at DESC LIMIT 1",
            (f"%{escaped}%",),
        ).fetchone()
    if row is None:
        return None
    term_, definition, title, chapter_index, model = row
    return GlossaryEntry(
        term=term_,
        definition=definition,
        book_title=title,
        chapter_index=int(chapter_index) if chapter_index is not None else None,
        model=model,
    )


def book_structuring_status(conn: sqlite3.Connection) -> list[BookStructuringStatus]:
    """One row per book: `"no"` (never queued -- no `book_chapters` rows at
    all, e.g. a PDF/`learn_text` book, or an EPUB not yet reached in the
    backlog), `"pending"` (has chapter rows, at least one still not
    `done`/`failed`), `"yes"` (every chapter row is `done`/`failed` and at
    least one is `done`). §7's small, independent addition -- a one-column
    view, not a new tool.
    """
    books = conn.execute("SELECT id, title FROM books ORDER BY id").fetchall()
    statuses = conn.execute("SELECT book_id, status FROM book_chapters").fetchall()
    by_book: dict[int, list[str]] = {}
    for book_id, status in statuses:
        by_book.setdefault(int(book_id), []).append(status)

    result: list[BookStructuringStatus] = []
    for book_id, title in books:
        chapter_statuses = by_book.get(int(book_id))
        if not chapter_statuses:
            structured = "no"
        elif any(s == "pending" for s in chapter_statuses):
            structured = "pending"
        elif any(s == "done" for s in chapter_statuses):
            structured = "yes"
        else:
            # Every chapter attempted and every one failed.
            structured = "yes"
        result.append(BookStructuringStatus(title=title, structured=structured))
    return result
