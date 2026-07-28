"""library-parse MCP server: ingest a book into the shared knowledge store.

No network beyond the one hardcoded Ollama call for embeddings. Runs on
whatever machine has Ollama (the MSI, not the Pi) -- see
docs/architecture/library-mcp.md.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from library_mcp.audit import AuditLog, Event
from library_mcp.config import ParsePolicy, PolicyError, load_parse_policy, policy_path_from_env
from library_mcp.embedding import EmbeddingClient, EmbeddingError
from library_mcp.parser import (
    ParseError,
    TextBlock,
    chunk_blocks,
    detect_doc_type,
    extract,
    extract_title,
)
from library_mcp.runtime import audit_from_env, build_app, serve
from library_mcp.store import (
    DuplicateBookError,
    add_book,
    add_chunk,
    book_count,
    chunk_count,
    delete_book,
    find_book_by_hash,
    find_books_by_title_substring,
    list_book_statuses,
    mark_book_status,
    open_store,
    update_book_progress,
)


@dataclass(frozen=True, slots=True)
class _Deps:
    policy: ParsePolicy
    audit: AuditLog


def _resolve_within(root: Path, file_name: str) -> Path | None:
    """Resolve a caller-given name against one root, refusing any escape.

    `file_name` is treated as a name (or relative path) inside `root` only --
    never as an absolute path, and never allowed to resolve outside it via
    `..`. This is the one caller-controlled input this server has, so it's
    the one thing worth being paranoid about. Returns None (not an error) if
    the name simply doesn't exist under this particular root, so callers can
    try the next one.
    """
    root = root.resolve()
    candidate = (root / file_name).resolve()
    if root not in candidate.parents and candidate != root:
        return None
    return candidate if candidate.is_file() else None


_DOC_PREFIX_RE = re.compile(r"^doc_[0-9a-f]+_", re.IGNORECASE)
_SEPARATOR_RE = re.compile(r"[ _,]+")


def _normalize_file_name(name: str) -> str:
    """Strip Hermes' doc_<hash>_ prefix and fold separator/case variants.

    A real, repeated failure mode: given the exact filename in a "[The user
    sent a document: ...]" message, the model instead reconstructs one from
    memory of the title -- consistently dropping the doc_<hash>_ prefix and
    normalizing punctuation (commas vs. underscores, doubled underscores).
    Verified against three real attempts, 2026-07-28, all missing exactly
    the prefix. This makes that specific, observed mismatch resolve anyway,
    deterministically, rather than hoping another prompt tweak sticks --
    the tool-calling reliability problem this project keeps running into is
    not something prompting alone has fixed so far.
    """
    name = _DOC_PREFIX_RE.sub("", name)
    return _SEPARATOR_RE.sub("_", name.lower()).strip("_")


def _fuzzy_find_within(root: Path, file_name: str) -> Path | None:
    """Normalized-name match against real files in `root`.

    Only returns a match when exactly one file normalizes to the same name --
    an ambiguous match is refused rather than guessed, consistent with this
    project's "don't invent certainty" principle.
    """
    target = _normalize_file_name(file_name)
    if not target or not root.is_dir():
        return None
    candidates = [
        entry
        for entry in root.iterdir()
        if entry.is_file() and _normalize_file_name(entry.name) == target
    ]
    return candidates[0] if len(candidates) == 1 else None


def _resolve_source(policy: ParsePolicy, file_name: str) -> tuple[Path, bool]:
    """Find `file_name` in the inbox, then the Telegram document cache.

    Two roots because books arrive two ways: placed directly in the inbox
    (the original, curated path), or sent as a Telegram attachment, which
    Hermes saves to its own generic cache -- a real gap found 2026-07-28
    when a sent PDF was invisible to `learn` because only inbox_path was
    mounted. Inbox is tried first since it's the deliberate, curated source.

    Exact matches are tried in both roots before falling back to fuzzy
    matching in either, so an exact match is never shadowed by a fuzzy one.
    Returns (path, was_fuzzy) so callers can note when forgiving matching
    was actually used.
    """
    roots = [policy.inbox_path]
    if policy.documents_path is not None:
        roots.append(policy.documents_path)

    for root in roots:
        found = _resolve_within(root, file_name)
        if found is not None:
            return found, False

    for root in roots:
        found = _fuzzy_find_within(root, file_name)
        if found is not None:
            return found, True

    msg = f"'{file_name}' was not found in the inbox or recent documents"
    raise ParseError(msg)


async def _ingest_in_background(deps: _Deps, title: str, source: str, content_hash: str, blocks: list[TextBlock]) -> None:
    """Parse-to-chunks is already done by the caller; this does the slow part
    (embedding, one HTTP call per chunk) off the tool-call path.

    Runs detached (fire-and-forget via asyncio.create_task), so every
    failure mode has to be caught and recorded here -- nothing propagates
    back to a caller that already got its response. Progress is written to
    the books table as it goes, so list_learned reflects real state rather
    than the caller having to guess whether a long ingestion finished.
    """
    policy = deps.policy
    chunks = chunk_blocks(blocks, policy.chunk_chars, policy.chunk_overlap_chars)
    embedder = EmbeddingClient(policy.ollama_base_url, policy.embedding_model, policy.embed_timeout_seconds)
    conn = open_store(policy.db_path)
    doc_type = detect_doc_type(blocks, Path(source).suffix.lower() or None)

    try:
        book_id = add_book(
            conn,
            title=title,
            source_path=source,
            ingested_at=datetime.now(UTC).isoformat(),
            content_hash=content_hash,
            status="embedding",
            total_chunks=len(chunks),
            doc_type=doc_type,
        )
    except DuplicateBookError as exc:
        # Race: another learn() call for the same content finished first,
        # between this call's earlier duplicate check and now.
        deps.audit.write(Event.INGEST_DENIED, detail=title, reason=str(exc))
        return

    embedded = 0
    try:
        for i, chunk in enumerate(chunks):
            try:
                vector = await embedder.embed(chunk.text)
            except EmbeddingError as exc:
                deps.audit.write(Event.EMBED_FAILED, detail=title, reason=str(exc), chunk_index=i)
                continue
            add_chunk(conn, book_id=book_id, chunk_index=i, section=chunk.section, text=chunk.text, embedding=vector)
            embedded += 1
            update_book_progress(conn, book_id, embedded)
    except Exception as exc:  # noqa: BLE001 - fire-and-forget task, nothing else will see this
        mark_book_status(conn, book_id, "failed")
        deps.audit.write(Event.INGEST_FAILED, detail=title, reason=f"unexpected error: {exc}")
        return

    mark_book_status(conn, book_id, "done" if embedded > 0 else "failed")
    deps.audit.write(
        Event.INGESTED, detail=title, source=source, chunks_total=len(chunks), chunks_embedded=embedded
    )


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _learn(deps: _Deps, file_name: str) -> str:
    policy = deps.policy
    try:
        path, was_fuzzy = _resolve_source(policy, file_name)
    except ParseError as exc:
        deps.audit.write(Event.INGEST_DENIED, detail=file_name, reason=str(exc))
        return f"Refused: {exc}"
    if was_fuzzy:
        deps.audit.write(Event.FUZZY_MATCHED, detail=file_name, resolved_to=path.name)

    if path.suffix.lower() not in policy.allowed_extensions:
        deps.audit.write(Event.INGEST_DENIED, detail=file_name, reason="extension not allowed")
        return f"'{file_name}': only {policy.allowed_extensions} are supported."

    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > policy.max_file_mb:
        deps.audit.write(Event.INGEST_DENIED, detail=file_name, reason=f"{size_mb:.1f}MB over limit")
        return f"'{file_name}' is {size_mb:.1f}MB, over the {policy.max_file_mb}MB limit."

    content_hash = _hash_bytes(path.read_bytes())
    conn = open_store(policy.db_path)
    existing = find_book_by_hash(conn, content_hash)
    if existing is not None:
        deps.audit.write(Event.INGEST_DENIED, detail=file_name, reason=f"duplicate of {existing!r}")
        return f"You already have this book: '{existing}'."

    try:
        blocks = extract(path)
    except ParseError as exc:
        deps.audit.write(Event.INGEST_FAILED, detail=file_name, reason=str(exc))
        return f"Could not learn '{file_name}': {exc}"

    # Real metadata title beats the filename when the file has one -- a
    # numeric or otherwise meaningless filename (e.g. "1234567.pdf") is
    # exactly the case this matters for.
    title = extract_title(path) or path.stem
    asyncio.create_task(_ingest_in_background(deps, title, str(path), content_hash, blocks))
    page_or_section_count = len(blocks)
    return (
        f"Started learning '{title}' ({page_or_section_count} section(s)) in the background -- "
        "this can take a few minutes for a long book. Check list_learned for progress, or just "
        "ask ask_library later; chunks become searchable as they finish embedding."
    )


def _learn_text(deps: _Deps, title: str, text: str, source_url: str) -> str:
    """Ingest already-extracted text (e.g. a fetched web page), not a file.

    This is the data path for the sandbox-fetch approval-gated link-following
    extension: sandbox-fetch runs on the Pi and returns fetched text to the
    frontier model as a tool result; the frontier passes that text here
    rather than there being any shared filesystem between the Pi and the
    MSI (where this server and Ollama live). See docs/architecture/library-mcp.md.
    """
    policy = deps.policy
    title = title.strip()
    if not title:
        return "Refused: a title is required."
    if not text.strip():
        deps.audit.write(Event.INGEST_DENIED, detail=title, reason="empty text")
        return f"Refused: no text content given for '{title}'."
    max_chars = policy.max_file_mb * 1024 * 1024  # reuse the same size budget as file uploads
    if len(text) > max_chars:
        deps.audit.write(Event.INGEST_DENIED, detail=title, reason=f"{len(text)} chars over limit")
        return f"'{title}' is {len(text)} chars, over the ~{max_chars} char limit."

    content_hash = _hash_bytes(text.encode("utf-8"))
    conn = open_store(policy.db_path)
    existing = find_book_by_hash(conn, content_hash)
    if existing is not None:
        deps.audit.write(Event.INGEST_DENIED, detail=title, reason=f"duplicate of {existing!r}")
        return f"You already have this content: '{existing}'."

    blocks = [TextBlock(section=None, text=text)]
    asyncio.create_task(
        _ingest_in_background(deps, title, source_url or "(no source url given)", content_hash, blocks)
    )
    return f"Started learning '{title}' in the background."


def _forget(deps: _Deps, title: str) -> str:
    """Remove a learned book/text (and its chunks) by title, or part of one.

    Refuses on zero or multiple matches rather than guessing which book was
    meant -- deleting the wrong one is unrecoverable, so this is exactly the
    kind of ambiguity worth refusing back to the caller, same principle as
    _fuzzy_find_within's exact-match-only-when-unambiguous rule.
    """
    title = title.strip()
    if not title:
        return "Refused: give at least part of a title to forget."
    conn = open_store(deps.policy.db_path)
    matches = find_books_by_title_substring(conn, title)
    if not matches:
        deps.audit.write(Event.DELETE_DENIED, detail=title, reason="no match")
        return f"No learned book matches '{title}'."
    if len(matches) > 1:
        deps.audit.write(Event.DELETE_DENIED, detail=title, reason=f"{len(matches)} matches, ambiguous")
        titles = "\n".join(f"- {t}" for _, t in matches)
        return f"'{title}' matches more than one book -- be more specific:\n{titles}"
    book_id, matched_title = matches[0]
    delete_book(conn, book_id)
    deps.audit.write(Event.DELETED, detail=matched_title)
    return f"Forgot '{matched_title}'."


def build_server(policy: ParsePolicy, audit: AuditLog) -> FastMCP:
    deps = _Deps(policy=policy, audit=audit)
    app = build_app("library-parse")

    @app.tool(
        description=(
            "Learn a book (PDF or EPUB) by file name -- checks both the shared inbox directory "
            "and recently received chat attachments (e.g. a PDF the user just sent on Telegram). "
            "Chunks and embeds it into the knowledge base in the background (returns immediately; "
            "a long book can take minutes to fully embed) so ask_library can search it later. "
            "Refuses if this exact content is already in the knowledge base. "
            "Give just the file name (e.g. 'networking-with-python.pdf'), not a full path. "
            "Note: this is a normal tool call, not the '/learn' slash command -- that command "
            "extracts a reusable skill from the conversation and is unrelated to books."
        )
    )
    def learn(file_name: str) -> str:
        return _learn(deps, file_name)

    @app.tool(
        description=(
            "Learn from already-fetched text (e.g. a web page returned by sandbox_fetch), "
            "not a file. Use this to add a fetched link's content to the knowledge base -- "
            "give it the title to file this under, the fetched text itself, and the source URL. "
            "Runs in the background like learn(); refuses if this exact content is already known."
        )
    )
    def learn_text(title: str, text: str, source_url: str = "") -> str:
        return _learn_text(deps, title, text, source_url)

    @app.tool(
        description=(
            "Remove a learned book/text from the knowledge base by title (or part of one) -- "
            "deletes it and its chunks permanently, so it will no longer show in list_learned "
            "or be found by ask_library. Refuses if the given title matches more than one book "
            "or no book at all, rather than guessing."
        )
    )
    def forget(title: str) -> str:
        return _forget(deps, title)

    @app.tool(
        description=(
            "List every book in the knowledge base, including ones still being embedded "
            "in the background, with progress."
        )
    )
    def list_learned() -> str:
        conn = open_store(deps.policy.db_path)
        statuses = list_book_statuses(conn)
        deps.audit.write(Event.LISTED, books=len(statuses), chunks=chunk_count(conn))
        if not statuses:
            return "No books learned yet."
        lines = []
        for b in statuses:
            doc_type_tag = f"[{b.doc_type}] " if b.doc_type else ""
            if b.status == "embedding":
                lines.append(
                    f"- {doc_type_tag}{b.title} (embedding: {b.embedded_chunks}/{b.total_chunks} chunks)"
                )
            elif b.status == "failed":
                lines.append(f"- {doc_type_tag}{b.title} (failed -- check the audit log)")
            else:
                lines.append(f"- {doc_type_tag}{b.title} ({b.embedded_chunks} chunks)")
        return f"{book_count(conn)} book(s), {chunk_count(conn)} chunk(s) total:\n" + "\n".join(lines)

    return app


def main() -> None:
    try:
        policy = load_parse_policy(policy_path_from_env())
    except PolicyError as exc:
        print(f"library-parse: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    serve(build_server(policy, audit_from_env()))


if __name__ == "__main__":
    main()
