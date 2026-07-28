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
from library_mcp.parser import ParseError, chunk_blocks, extract
from library_mcp.runtime import audit_from_env, build_app, serve
from library_mcp.store import add_book, add_chunk, book_count, chunk_count, commit, list_books, open_store


@dataclass(frozen=True, slots=True)
class _Deps:
    policy: ParsePolicy
    audit: AuditLog


def _resolve_in_inbox(policy: ParsePolicy, file_name: str) -> Path:
    """Resolve a caller-given name against the inbox, refusing any escape.

    `file_name` is treated as a name (or relative path) inside the inbox
    only -- never as an absolute path, and never allowed to resolve outside
    it via `..`. This is the one caller-controlled input this server has, so
    it's the one thing worth being paranoid about.
    """
    candidate = (policy.inbox_path / file_name).resolve()
    inbox = policy.inbox_path.resolve()
    if inbox not in candidate.parents and candidate != inbox:
        msg = f"'{file_name}' resolves outside the inbox directory"
        raise ParseError(msg)
    return candidate


def _learn(deps: _Deps, file_name: str) -> str:
    policy = deps.policy
    try:
        path = _resolve_in_inbox(policy, file_name)
    except ParseError as exc:
        deps.audit.write(Event.INGEST_DENIED, detail=file_name, reason=str(exc))
        return f"Refused: {exc}"

    if not path.is_file():
        deps.audit.write(Event.INGEST_DENIED, detail=file_name, reason="not a file")
        return f"'{file_name}' is not a file in the inbox."

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

    chunks = chunk_blocks(blocks, policy.chunk_chars, policy.chunk_overlap_chars)
    embedder = EmbeddingClient(policy.ollama_base_url, policy.embedding_model, policy.embed_timeout_seconds)

    conn = open_store(policy.db_path)
    title = path.stem
    book_id = add_book(conn, title=title, source_path=str(path), ingested_at=datetime.now(UTC).isoformat())

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
        Event.INGESTED, detail=title, source=str(path), chunks_total=len(chunks), chunks_embedded=embedded
    )
    if embedded < len(chunks):
        return (
            f"Learned '{title}': {embedded}/{len(chunks)} chunks embedded "
            f"({len(chunks) - embedded} failed -- check the audit log)."
        )
    return f"Learned '{title}': {embedded} chunks from {len(blocks)} section(s)."


def build_server(policy: ParsePolicy, audit: AuditLog) -> FastMCP:
    deps = _Deps(policy=policy, audit=audit)
    app = build_app("library-parse")

    @app.tool(
        description=(
            "Learn a book (PDF or EPUB) by file name, from the shared inbox directory. "
            "Chunks and embeds it into the knowledge base so ask_library can search it later. "
            "Give just the file name (e.g. 'networking-with-python.pdf'), not a full path."
        )
    )
    def learn(file_name: str) -> str:
        return _learn(deps, file_name)

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
