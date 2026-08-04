"""Book Q&A escalation flow (docs/planning/book-qa-escalation-flow.md,
§7 build plan steps 1-5) -- mirrors test_ask_library.py's mocking style.

Covers: `_clean_question()` (Stage 1, pure string function), `AskOutcome`/
`_ask_structured()` (Stage 2's structured outcome, one test per branch),
and `_ask()`'s escalation-branch wording (Stage 2->3 trigger + interim
reply, §3-4) -- including the distinction that matters most: the two
knowledge-gap outcomes get the new honest wording, the two infra-failure
outcomes keep the original text unchanged.
"""

from pathlib import Path
from unittest.mock import AsyncMock, patch

from library_mcp.audit import AuditLog
from library_mcp.config import KeeperPolicy
from library_mcp.embedding import EmbeddingError
from library_mcp.external_sources import ExternalResult, ExternalSourceError
from library_mcp.keeper_model import AnswerDecision, ReasoningError, SearchDecision
from library_mcp.servers.keeper_server import (
    _NO_MATCHES_INTERIM_REPLY,
    _REPEATED_QUERY_INTERIM_REPLY,
    AskOutcome,
    _ask,
    _ask_structured,
    _clean_question,
    _Deps,
)
from library_mcp.store import add_book, add_chunk, commit, open_store


def _deps(tmp_path: Path) -> _Deps:
    policy = KeeperPolicy(ollama_base_url="http://unused:11434", db_path=tmp_path / "knowledge.db")
    return _Deps(policy=policy, audit=AuditLog(tmp_path / "audit.jsonl"))


# ---------------------------------------------------------------------------
# Stage 1: _clean_question -- pure string function, no model, no DB
# ---------------------------------------------------------------------------


def test_clean_question_strips_the_exact_quiet_mode_wrapper() -> None:
    wrapped = (
        "[Quiet mode is ON. Answer only the message below, as briefly as "
        "possible. Do not ask a follow-up question. Do not bring up earlier "
        "topics or add commentary beyond what was asked.]\n\n"
        "what does the book say about stoic acceptance?"
    )
    assert _clean_question(wrapped) == "what does the book say about stoic acceptance?"


def test_clean_question_normalizes_whitespace_and_newlines() -> None:
    messy = "what   is\n\nthe   role\tof   virtue?"
    assert _clean_question(messy) == "what is the role of virtue?"


def test_clean_question_trims_surrounding_quote_and_punctuation_artifacts() -> None:
    assert _clean_question('"what is courage?"') == "what is courage?"
    assert _clean_question("'what is courage?'") == "what is courage?"


def test_clean_question_does_not_mangle_a_real_question_mentioning_quiet() -> None:
    # The overly-broad-strip trap named explicitly in the brief: a
    # legitimate question that happens to contain a word similar to the
    # wrapper text (here, "quiet") must pass through untouched, since the
    # strip list matches the literal wrapper block, not a loose keyword.
    real_question = "what does the book say about finding a quiet mind through meditation?"
    assert _clean_question(real_question) == real_question


def test_clean_question_leaves_a_clean_question_unchanged() -> None:
    assert _clean_question("what is the role of virtue in stoicism?") == (
        "what is the role of virtue in stoicism?"
    )


# ---------------------------------------------------------------------------
# Stage 2: AskOutcome / _ask_structured -- one test per branch
# ---------------------------------------------------------------------------


async def test_ask_structured_answered_with_real_results(tmp_path: Path) -> None:
    deps = _deps(tmp_path)
    conn = open_store(deps.policy.db_path)
    book_id = add_book(conn, "Some Book", "/inbox/a.pdf", "2026-01-01")
    add_chunk(conn, book_id, 0, "p1", "relevant passage", [1.0, 0.0])
    commit(conn)

    with (
        patch("library_mcp.servers.keeper_server.EmbeddingClient") as mock_embed,
        patch("library_mcp.servers.keeper_server.ReasoningClient") as mock_reason,
    ):
        mock_embed.return_value.embed = AsyncMock(return_value=[1.0, 0.0])
        mock_reason.return_value.decide = AsyncMock(
            return_value=AnswerDecision(text="Here's the answer.")
        )

        outcome = await _ask_structured(deps, "a real question")

    assert outcome == AskOutcome(text="Here's the answer.", status="answered", gap_id=None)


async def test_ask_structured_no_matches_sets_status_and_gap_id(tmp_path: Path) -> None:
    deps = _deps(tmp_path)
    with (
        patch("library_mcp.servers.keeper_server.EmbeddingClient") as mock_embed,
        patch("library_mcp.servers.keeper_server.ReasoningClient") as mock_reason,
    ):
        mock_embed.return_value.embed = AsyncMock(return_value=[1.0, 0.0])
        mock_reason.return_value.decide = AsyncMock(
            return_value=AnswerDecision(text="The passages don't cover this.")
        )

        outcome = await _ask_structured(deps, "what does the book say about zorbnaxx?")

    assert outcome.status == "no_matches"
    assert outcome.text == "The passages don't cover this."
    assert isinstance(outcome.gap_id, int)


async def test_ask_structured_repeated_query_sets_status_and_gap_id(tmp_path: Path) -> None:
    deps = _deps(tmp_path)
    conn = open_store(deps.policy.db_path)
    book_id = add_book(conn, "Some Book", "/inbox/a.pdf", "2026-01-01")
    add_chunk(conn, book_id, 0, "p1", "somewhat related passage", [1.0, 0.0])
    commit(conn)

    with (
        patch("library_mcp.servers.keeper_server.EmbeddingClient") as mock_embed,
        patch("library_mcp.servers.keeper_server.ReasoningClient") as mock_reason,
    ):
        mock_embed.return_value.embed = AsyncMock(return_value=[1.0, 0.0])
        mock_reason.return_value.decide = AsyncMock(
            return_value=SearchDecision(query="still looking")
        )

        outcome = await _ask_structured(deps, "a hard question")

    assert outcome.status == "repeated_query_no_answer"
    assert isinstance(outcome.gap_id, int)
    assert "couldn't settle on a confident answer" in outcome.text


async def test_ask_structured_embed_failed_has_no_gap_id(tmp_path: Path) -> None:
    deps = _deps(tmp_path)
    with patch("library_mcp.servers.keeper_server.EmbeddingClient") as mock_embed:
        mock_embed.return_value.embed = AsyncMock(side_effect=EmbeddingError("boom"))

        outcome = await _ask_structured(deps, "a real question")

    assert outcome.status == "embed_failed"
    assert outcome.gap_id is None
    assert outcome.text == "I couldn't search the library right now (embedding call failed)."


async def test_ask_structured_reasoning_failed_has_no_gap_id(tmp_path: Path) -> None:
    deps = _deps(tmp_path)
    with (
        patch("library_mcp.servers.keeper_server.EmbeddingClient") as mock_embed,
        patch("library_mcp.servers.keeper_server.ReasoningClient") as mock_reason,
    ):
        mock_embed.return_value.embed = AsyncMock(return_value=[1.0, 0.0])
        mock_reason.return_value.decide = AsyncMock(side_effect=ReasoningError("boom"))

        outcome = await _ask_structured(deps, "a real question")

    assert outcome.status == "reasoning_failed"
    assert outcome.gap_id is None
    assert outcome.text == "I couldn't reason over the retrieved passages right now."


# ---------------------------------------------------------------------------
# Stage 2->3 escalation branching, exercised through the thin _ask() wrapper
# ---------------------------------------------------------------------------


async def test_ask_returns_interim_reply_for_no_matches(tmp_path: Path) -> None:
    # Both the book search AND the live web fallback come up empty here --
    # the honest interim reply is what's left. See the *_web_fallback tests
    # below for the case where the web search actually finds something.
    deps = _deps(tmp_path)
    with (
        patch("library_mcp.servers.keeper_server.EmbeddingClient") as mock_embed,
        patch("library_mcp.servers.keeper_server.ReasoningClient") as mock_reason,
        patch("library_mcp.servers.keeper_server.DdgsSource") as mock_ddgs,
    ):
        mock_embed.return_value.embed = AsyncMock(return_value=[1.0, 0.0])
        mock_reason.return_value.decide = AsyncMock(
            return_value=AnswerDecision(text="The passages don't cover this.")
        )
        mock_ddgs.return_value.lookup = AsyncMock(return_value=None)

        result = await _ask(deps, "what does the book say about zorbnaxx?")

    assert result == _NO_MATCHES_INTERIM_REPLY
    # Not the old bare fallback text -- the model's own raw answer text must
    # never reach the caller for this branch.
    assert result != "The passages don't cover this."


async def test_ask_returns_interim_reply_for_repeated_query(tmp_path: Path) -> None:
    deps = _deps(tmp_path)
    conn = open_store(deps.policy.db_path)
    book_id = add_book(conn, "Some Book", "/inbox/a.pdf", "2026-01-01")
    add_chunk(conn, book_id, 0, "p1", "somewhat related passage", [1.0, 0.0])
    commit(conn)

    with (
        patch("library_mcp.servers.keeper_server.EmbeddingClient") as mock_embed,
        patch("library_mcp.servers.keeper_server.ReasoningClient") as mock_reason,
        patch("library_mcp.servers.keeper_server.DdgsSource") as mock_ddgs,
    ):
        mock_embed.return_value.embed = AsyncMock(return_value=[1.0, 0.0])
        mock_reason.return_value.decide = AsyncMock(
            return_value=SearchDecision(query="still looking")
        )
        mock_ddgs.return_value.lookup = AsyncMock(return_value=None)

        result = await _ask(deps, "a hard question")

    assert result == _REPEATED_QUERY_INTERIM_REPLY
    assert "Closest passages found" not in result


# ---------------------------------------------------------------------------
# Live web fallback (new): tried within the same turn when the book search
# comes up empty, before falling back to the honest interim reply.
# ---------------------------------------------------------------------------


async def test_ask_uses_live_web_fallback_when_book_search_finds_nothing(tmp_path: Path) -> None:
    deps = _deps(tmp_path)
    with (
        patch("library_mcp.servers.keeper_server.EmbeddingClient") as mock_embed,
        patch("library_mcp.servers.keeper_server.ReasoningClient") as mock_reason,
        patch("library_mcp.servers.keeper_server.DdgsSource") as mock_ddgs,
    ):
        mock_embed.return_value.embed = AsyncMock(return_value=[1.0, 0.0])
        mock_reason.return_value.decide = AsyncMock(
            return_value=AnswerDecision(text="The passages don't cover this.")
        )
        mock_ddgs.return_value.lookup = AsyncMock(
            return_value=ExternalResult(
                source="web",
                title="what does the book say about zorbnaxx?",
                url="https://example.com/zorbnaxx",
                extract="Zorbnaxx is a fictional example term.",
                fetched_at="2026-08-04T00:00:00+00:00",
            )
        )

        result = await _ask(deps, "what does the book say about zorbnaxx?")

    # Never silently returns the interim "let me dig deeper" reply when a
    # real web answer was found -- and never claims book-grounding for it.
    assert result != _NO_MATCHES_INTERIM_REPLY
    assert "doesn't cover this" in result
    assert "Zorbnaxx is a fictional example term." in result
    assert "https://example.com/zorbnaxx" in result


async def test_ask_falls_back_to_interim_reply_when_web_search_errors(tmp_path: Path) -> None:
    # A broken web search (timeout, worker crash, etc.) must fail open to
    # the existing honest interim reply -- never surface as an error to the
    # user, never block the answer path that already worked before this
    # feature existed.
    deps = _deps(tmp_path)
    with (
        patch("library_mcp.servers.keeper_server.EmbeddingClient") as mock_embed,
        patch("library_mcp.servers.keeper_server.ReasoningClient") as mock_reason,
        patch("library_mcp.servers.keeper_server.DdgsSource") as mock_ddgs,
    ):
        mock_embed.return_value.embed = AsyncMock(return_value=[1.0, 0.0])
        mock_reason.return_value.decide = AsyncMock(
            return_value=AnswerDecision(text="The passages don't cover this.")
        )
        mock_ddgs.return_value.lookup = AsyncMock(
            side_effect=ExternalSourceError("web search timed out after 20.0s")
        )

        result = await _ask(deps, "what does the book say about zorbnaxx?")

    assert result == _NO_MATCHES_INTERIM_REPLY


async def test_ask_does_not_use_live_web_fallback_when_the_book_answers(tmp_path: Path) -> None:
    # A real book answer must never be second-guessed or replaced by a web
    # search -- the fallback path is only reachable for no_matches/
    # repeated_query_no_answer, never for a genuine "answered" outcome.
    deps = _deps(tmp_path)
    conn = open_store(deps.policy.db_path)
    book_id = add_book(conn, "Some Book", "/inbox/a.pdf", "2026-01-01")
    add_chunk(conn, book_id, 0, "p1", "relevant passage", [1.0, 0.0])
    commit(conn)

    with (
        patch("library_mcp.servers.keeper_server.EmbeddingClient") as mock_embed,
        patch("library_mcp.servers.keeper_server.ReasoningClient") as mock_reason,
        patch("library_mcp.servers.keeper_server.DdgsSource") as mock_ddgs,
    ):
        mock_embed.return_value.embed = AsyncMock(return_value=[1.0, 0.0])
        mock_reason.return_value.decide = AsyncMock(
            return_value=AnswerDecision(text="Here's the answer.")
        )
        mock_ddgs.return_value.lookup = AsyncMock(
            side_effect=AssertionError("DdgsSource.lookup must not be called when the book answers")
        )

        result = await _ask(deps, "a real question")

    assert result == "Here's the answer."


async def test_ask_does_not_reframe_embed_failure_as_a_knowledge_gap(tmp_path: Path) -> None:
    # The distinction most likely to get accidentally collapsed: an infra
    # failure must keep its original text unchanged, never the interim-reply
    # framing (which would incorrectly imply "the books don't have this").
    deps = _deps(tmp_path)
    with patch("library_mcp.servers.keeper_server.EmbeddingClient") as mock_embed:
        mock_embed.return_value.embed = AsyncMock(side_effect=EmbeddingError("boom"))

        result = await _ask(deps, "a real question")

    assert result == "I couldn't search the library right now (embedding call failed)."
    assert result != _NO_MATCHES_INTERIM_REPLY
    assert result != _REPEATED_QUERY_INTERIM_REPLY


async def test_ask_does_not_reframe_reasoning_failure_as_a_knowledge_gap(tmp_path: Path) -> None:
    deps = _deps(tmp_path)
    with (
        patch("library_mcp.servers.keeper_server.EmbeddingClient") as mock_embed,
        patch("library_mcp.servers.keeper_server.ReasoningClient") as mock_reason,
    ):
        mock_embed.return_value.embed = AsyncMock(return_value=[1.0, 0.0])
        mock_reason.return_value.decide = AsyncMock(side_effect=ReasoningError("boom"))

        result = await _ask(deps, "a real question")

    assert result == "I couldn't reason over the retrieved passages right now."
    assert result != _NO_MATCHES_INTERIM_REPLY
    assert result != _REPEATED_QUERY_INTERIM_REPLY


async def test_ask_returns_the_answer_unchanged_when_answered(tmp_path: Path) -> None:
    deps = _deps(tmp_path)
    conn = open_store(deps.policy.db_path)
    book_id = add_book(conn, "Some Book", "/inbox/a.pdf", "2026-01-01")
    add_chunk(conn, book_id, 0, "p1", "relevant passage", [1.0, 0.0])
    commit(conn)

    with (
        patch("library_mcp.servers.keeper_server.EmbeddingClient") as mock_embed,
        patch("library_mcp.servers.keeper_server.ReasoningClient") as mock_reason,
    ):
        mock_embed.return_value.embed = AsyncMock(return_value=[1.0, 0.0])
        mock_reason.return_value.decide = AsyncMock(
            return_value=AnswerDecision(text="Here's the answer.")
        )

        result = await _ask(deps, "a real question")

    assert result == "Here's the answer."


# ---------------------------------------------------------------------------
# ask_library's MCP tool contract: still question: str -> str, unchanged
# ---------------------------------------------------------------------------


async def test_ask_library_tool_still_returns_a_plain_string(tmp_path: Path) -> None:
    from library_mcp.servers.keeper_server import build_server

    policy = KeeperPolicy(ollama_base_url="http://unused:11434", db_path=tmp_path / "knowledge.db")
    server = build_server(policy, AuditLog(tmp_path / "audit.jsonl"))
    ask_library_fn = server._tool_manager._tools["ask_library"].fn  # type: ignore[attr-defined]

    conn = open_store(policy.db_path)
    book_id = add_book(conn, "Some Book", "/inbox/a.pdf", "2026-01-01")
    add_chunk(conn, book_id, 0, "p1", "relevant passage", [1.0, 0.0])
    commit(conn)

    with (
        patch("library_mcp.servers.keeper_server.EmbeddingClient") as mock_embed,
        patch("library_mcp.servers.keeper_server.ReasoningClient") as mock_reason,
    ):
        mock_embed.return_value.embed = AsyncMock(return_value=[1.0, 0.0])
        mock_reason.return_value.decide = AsyncMock(
            return_value=AnswerDecision(text="Here's the answer.")
        )

        result = await ask_library_fn(question="a real question")

    assert isinstance(result, str)
    assert result == "Here's the answer."
