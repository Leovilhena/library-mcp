"""library-parse MCP server: ingest a book into the shared knowledge store.

No network beyond the one hardcoded Ollama call for embeddings. Runs on
whatever machine has Ollama (the MSI, not the Pi) -- see
docs/architecture/library-mcp.md.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from library_mcp.audit import AuditLog, Event
from library_mcp.config import ParsePolicy, PolicyError, load_parse_policy, policy_path_from_env
from library_mcp.embedding import EmbeddingClient, EmbeddingError
from library_mcp.parser import ParseError, TextBlock, chunk_blocks, extract
from library_mcp.runtime import audit_from_env, build_app, serve
from library_mcp.store import add_book, add_chunk, book_count, chunk_count, commit, list_books, open_store


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


def _resolve_source(policy: ParsePolicy, file_name: str) -> Path:
    """Find `file_name` in the inbox, then the Telegram document cache.

    Two roots because books arrive two ways: placed directly in the inbox
    (the original, curated path), or sent as a Telegram attachment, which
    Hermes saves to its own generic cache -- a real gap found 2026-07-28
    when a sent PDF was invisible to `learn` because only inbox_path was
    mounted. Inbox is tried first since it's the deliberate, curated source.
    """
    found = _resolve_within(policy.inbox_path, file_name)
    if found is not None:
        return found
    if policy.documents_path is not None:
        found = _resolve_within(policy.documents_path, file_name)
        if found is not None:
            return found
    msg = f"'{file_name}' was not found in the inbox or recent documents"
    raise ParseError(msg)


def _ingest_blocks(deps: _Deps, title: str, source: str, blocks: list[TextBlock]) -> str:
    policy = deps.policy
    chunks = chunk_blocks(blocks, policy.chunk_chars, policy.chunk_overlap_chars)
    embedder = EmbeddingClient(policy.ollama_base_url, policy.embedding_model, policy.embed_timeout_seconds)

    conn = open_store(policy.db_path)
    book_id = add_book(conn, title=title, source_path=source, ingested_at=datetime.now(UTC).isoformat())

    embedded = 0
    for i, chunk in enumerate(chunks):
        try:
            vector = embedder.embed(chunk.text)
        except EmbeddingError as exc:
            deps.audit.write(Event.EMBED_FAILED, detail=title, reason=str(exc), chunk_index=i)
            continue
        add_chunk(conn, book_id=book_id, chunk_index=i, section=chunk.section, text=chunk.text, embedding=vector)
        embedded += 1
    commit(conn)

    deps.audit.write(
        Event.INGESTED, detail=title, source=source, chunks_total=len(chunks), chunks_embedded=embedded
    )
    if embedded < len(chunks):
        return (
            f"Learned '{title}': {embedded}/{len(chunks)} chunks embedded "
            f"({len(chunks) - embedded} failed -- check the audit log)."
        )
    return f"Learned '{title}': {embedded} chunks from {len(blocks)} section(s)."


def _learn(deps: _Deps, file_name: str) -> str:
    policy = deps.policy
    try:
        path = _resolve_source(policy, file_name)
    except ParseError as exc:
        deps.audit.write(Event.INGEST_DENIED, detail=file_name, reason=str(exc))
        return f"Refused: {exc}"

    if path.suffix.lower() not in policy.allowed_extensions:
        deps.audit.write(Event.INGEST_DENIED, detail=file_name, reason="extension not allowed")
        return f"'{file_name}': only {policy.allowed_extensions} are supported."

    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > policy.max_file_mb:
        deps.audit.write(Event.INGEST_DENIED, detail=file_name, reason=f"{size_mb:.1f}MB over limit")
        return f"'{file_name}' is {size_mb:.1f}MB, over the {policy.max_file_mb}MB limit."

    try:
        blocks = extract(path)
    except ParseError as exc:
        deps.audit.write(Event.INGEST_FAILED, detail=file_name, reason=str(exc))
        return f"Could not learn '{file_name}': {exc}"

    return _ingest_blocks(deps, title=path.stem, source=str(path), blocks=blocks)


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

    blocks = [TextBlock(section=None, text=text)]
    return _ingest_blocks(deps, title=title, source=source_url or "(no source url given)", blocks=blocks)


def build_server(policy: ParsePolicy, audit: AuditLog) -> FastMCP:
    deps = _Deps(policy=policy, audit=audit)
    app = build_app("library-parse")

    @app.tool(
        description=(
            "Learn a book (PDF or EPUB) by file name -- checks both the shared inbox directory "
            "and recently received chat attachments (e.g. a PDF the user just sent on Telegram). "
            "Chunks and embeds it into the knowledge base so ask_library can search it later. "
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
            "give it the title to file this under, the fetched text itself, and the source URL."
        )
    )
    def learn_text(title: str, text: str, source_url: str = "") -> str:
        return _learn_text(deps, title, text, source_url)

    @app.tool(description="List every book currently in the knowledge base.")
    def list_learned() -> str:
        conn = open_store(deps.policy.db_path)
        titles = list_books(conn)
        deps.audit.write(Event.LISTED, books=len(titles), chunks=chunk_count(conn))
        if not titles:
            return "No books learned yet."
        return f"{book_count(conn)} book(s), {chunk_count(conn)} chunk(s) total:\n" + "\n".join(
            f"- {t}" for t in titles
        )

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
