"""`_ask()`/`_format_context` book-structuring integration tests
(docs/planning/book-structuring.md §4, §9 build step 4) -- mirrors
test_ask_library.py's mocking style.

The requirement this file exists to prove, per the design doc's own §4:
the term-lookup pre-check and the labeled glossary block are strictly
ADDITIVE. Real chunk search must run every time, glossary hit or not."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

from library_mcp.audit import AuditLog
from library_mcp.config import KeeperPolicy
from library_mcp.keeper_model import AnswerDecision
from library_mcp.servers.keeper_server import _ask, _Deps, _format_context, _term_lookup_candidate
from library_mcp.store import (
    GlossaryEntry,
    SearchResult,
    add_book,
    add_chapter,
    add_chunk,
    add_glossary_term,
    commit,
    open_store,
)


def _deps(tmp_path: Path) -> _Deps:
    # Same construction as test_ask_library.py's own private helper --
    # duplicated rather than imported since tests/ has no __init__.py
    # (no package-relative imports between test modules).
    policy = KeeperPolicy(ollama_base_url="http://unused:11434", db_path=tmp_path / "knowledge.db")
    return _Deps(policy=policy, audit=AuditLog(tmp_path / "audit.jsonl"))


# ---------------------------------------------------------------------------
# _term_lookup_candidate: the deterministic pre-check heuristic
# ---------------------------------------------------------------------------


def test_term_lookup_candidate_matches_short_definition_questions() -> None:
    assert _term_lookup_candidate("what is recursion?") == "recursion"
    assert _term_lookup_candidate("What is a closure") == "closure"
    assert _term_lookup_candidate("define tail recursion") == "tail recursion"
    assert _term_lookup_candidate("what does asyncio mean?") == "asyncio"


def test_term_lookup_candidate_returns_none_for_longer_questions() -> None:
    # A broad multi-clause question that happens to start with "what is"
    # must NOT fire -- injecting a wrong glossary hint into a broad
    # question is more likely to mislead than help (§4).
    assert (
        _term_lookup_candidate(
            "what is the relationship between the author's views on courage "
            "and the earlier chapter on fear?"
        )
        is None
    )


def test_term_lookup_candidate_returns_none_for_unrelated_questions() -> None:
    assert _term_lookup_candidate("summarize chapter three for me") is None
    assert _term_lookup_candidate("list every value mentioned in the book") is None


# ---------------------------------------------------------------------------
# _format_context: labeling and additivity
# ---------------------------------------------------------------------------


def test_format_context_labels_a_glossary_entry_distinctly_from_chunks() -> None:
    glossary_entry = GlossaryEntry(
        term="recursion",
        definition="A function that calls itself.",
        book_title="A Book",
        chapter_index=0,
        model="deepseek-r1:8b",
    )
    results = [SearchResult(book_title="A Book", section="ch1", text="raw chunk text", score=0.9)]

    context = _format_context(results, max_chars=6000, glossary_entry=glossary_entry)

    assert "[Glossary entry -- Pythia's own synthesis, not verbatim book text -- A Book]" in context
    assert "recursion: A function that calls itself." in context
    assert "[A Book -- ch1]" in context
    assert "raw chunk text" in context
    # The glossary block must appear as its own distinct block, never
    # merged into a chunk's own bracketed header.
    assert "[A Book -- ch1]\nraw chunk text" in context


def test_format_context_with_no_glossary_entry_is_unchanged() -> None:
    results = [SearchResult(book_title="A Book", section="ch1", text="raw chunk text", score=0.9)]
    context = _format_context(results, max_chars=6000, glossary_entry=None)
    assert "Glossary entry" not in context
    assert "[A Book -- ch1]\nraw chunk text" in context


# ---------------------------------------------------------------------------
# _ask(): end-to-end -- glossary hit is additive, chunk search still runs
# ---------------------------------------------------------------------------


async def test_ask_includes_glossary_context_alongside_normal_chunk_search(
    tmp_path: Path,
) -> None:
    deps = _deps(tmp_path)
    conn = open_store(deps.policy.db_path)
    book_id = add_book(conn, "Recursion Book", "/inbox/a.epub", "2026-01-01")
    chapter_id = add_chapter(conn, book_id, 0, "chap1.xhtml")
    add_glossary_term(
        conn,
        book_id,
        chapter_id,
        "recursion",
        "A function that calls itself to solve smaller instances of a problem.",
        "deepseek-r1:8b",
        "2026-07-30T00:00:00",
    )
    # Real chunk that a normal search would find -- distinct text from the
    # glossary definition, so the test can tell them apart in the captured
    # context.
    add_chunk(conn, book_id, 0, "chap1.xhtml", "raw verbatim passage about recursion", [1.0, 0.0])
    commit(conn)

    captured_context: dict[str, str] = {}

    async def _fake_decide(_question, context, _remaining):
        captured_context["value"] = context
        return AnswerDecision(text="answered")

    with (
        patch("library_mcp.servers.keeper_server.EmbeddingClient") as mock_embed,
        patch("library_mcp.servers.keeper_server.ReasoningClient") as mock_reason,
    ):
        mock_embed.return_value.embed = AsyncMock(return_value=[1.0, 0.0])
        mock_reason.return_value.decide = AsyncMock(side_effect=_fake_decide)

        result = await _ask(deps, "what is recursion?")

    assert result == "answered"
    context = captured_context["value"]
    # Both present -- additive, not a substitute (§4's explicit requirement).
    assert "Glossary entry" in context
    assert "raw verbatim passage about recursion" in context


async def test_ask_runs_normal_search_even_when_no_glossary_hit_exists(
    tmp_path: Path,
) -> None:
    # A term-shaped question with NO matching glossary row must still run
    # real chunk search -- the pre-check firing or not never gates search.
    deps = _deps(tmp_path)
    conn = open_store(deps.policy.db_path)
    book_id = add_book(conn, "Some Book", "/inbox/a.epub", "2026-01-01")
    add_chunk(conn, book_id, 0, "chap1.xhtml", "unrelated passage text", [1.0, 0.0])
    commit(conn)

    captured_context: dict[str, str] = {}

    async def _fake_decide(_question, context, _remaining):
        captured_context["value"] = context
        return AnswerDecision(text="answered")

    with (
        patch("library_mcp.servers.keeper_server.EmbeddingClient") as mock_embed,
        patch("library_mcp.servers.keeper_server.ReasoningClient") as mock_reason,
    ):
        mock_embed.return_value.embed = AsyncMock(return_value=[1.0, 0.0])
        mock_reason.return_value.decide = AsyncMock(side_effect=_fake_decide)

        await _ask(deps, "what is quantumfoo?")

    context = captured_context["value"]
    assert "Glossary entry" not in context
    assert "unrelated passage text" in context


async def test_ask_runs_normal_search_for_a_non_term_lookup_question(tmp_path: Path) -> None:
    # A question that doesn't even look like a term lookup must still get
    # normal search, unaffected by the pre-check's absence of a match.
    deps = _deps(tmp_path)
    conn = open_store(deps.policy.db_path)
    book_id = add_book(conn, "Some Book", "/inbox/a.epub", "2026-01-01")
    chapter_id = add_chapter(conn, book_id, 0, "chap1.xhtml")
    add_glossary_term(
        conn,
        book_id,
        chapter_id,
        "recursion",
        "irrelevant here",
        "deepseek-r1:8b",
        "2026-07-30T00:00:00",
    )
    add_chunk(conn, book_id, 0, "chap1.xhtml", "a broad passage of real content", [1.0, 0.0])
    commit(conn)

    captured_context: dict[str, str] = {}

    async def _fake_decide(_question, context, _remaining):
        captured_context["value"] = context
        return AnswerDecision(text="answered")

    with (
        patch("library_mcp.servers.keeper_server.EmbeddingClient") as mock_embed,
        patch("library_mcp.servers.keeper_server.ReasoningClient") as mock_reason,
    ):
        mock_embed.return_value.embed = AsyncMock(return_value=[1.0, 0.0])
        mock_reason.return_value.decide = AsyncMock(side_effect=_fake_decide)

        await _ask(deps, "summarize this book's overall argument for me")

    context = captured_context["value"]
    assert "Glossary entry" not in context
    assert "a broad passage of real content" in context
