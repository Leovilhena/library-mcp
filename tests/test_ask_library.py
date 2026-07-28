from pathlib import Path
from unittest.mock import AsyncMock, patch

from library_mcp.audit import AuditLog
from library_mcp.config import KeeperPolicy
from library_mcp.keeper_model import AnswerDecision, SearchDecision
from library_mcp.servers.keeper_server import _ask, _Deps
from library_mcp.store import add_book, add_chunk, commit, list_knowledge_gaps, open_store


def _deps(tmp_path: Path) -> _Deps:
    policy = KeeperPolicy(ollama_base_url="http://unused:11434", db_path=tmp_path / "knowledge.db")
    return _Deps(policy=policy, audit=AuditLog(tmp_path / "audit.jsonl"))


async def test_ask_library_logs_a_gap_when_nothing_matches(tmp_path: Path) -> None:
    # Real behavior, not the mock-past-it version: the keeper's own prompt
    # (keeper_model.py) explicitly tells the reasoner "you must respond with
    # ANSWER using whatever you have, even if it means saying the passages
    # don't cover it" -- so an empty-library question surfaces as a real
    # AnswerDecision with no results backing it, not a SearchDecision. This
    # is the actually-reachable "no_matches" path in production.
    deps = _deps(tmp_path)
    with (
        patch("library_mcp.servers.keeper_server.EmbeddingClient") as mock_embed,
        patch("library_mcp.servers.keeper_server.ReasoningClient") as mock_reason,
    ):
        mock_embed.return_value.embed = AsyncMock(return_value=[1.0, 0.0])
        mock_reason.return_value.decide = AsyncMock(
            return_value=AnswerDecision(text="The passages don't cover this.")
        )

        await _ask(deps, "what does the book say about zorbnaxx?")

    conn = open_store(deps.policy.db_path)
    gaps = list_knowledge_gaps(conn)
    assert len(gaps) == 1
    assert gaps[0].question == "what does the book say about zorbnaxx?"
    assert gaps[0].reason == "no_matches"


async def test_ask_library_does_not_log_a_gap_when_answered(tmp_path: Path) -> None:
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
    assert list_knowledge_gaps(open_store(deps.policy.db_path)) == []


async def test_ask_library_logs_repeated_query_reason(tmp_path: Path) -> None:
    # The other reachable gap path: the reasoner asks for the same search
    # query twice in a row, which breaks the loop early (rather than
    # spinning) before ever producing an AnswerDecision. Real matches exist
    # here, so this is a distinct reason from "no_matches".
    #
    # Note what this is NOT testing: naturally exhausting `max_searches`
    # without an answer. `_parse_decision` (keeper_model.py) always returns
    # AnswerDecision once `remaining_searches` hits 0 -- the prompt tells the
    # model to answer regardless on its last attempt -- so the for loop can
    # never fall through to its "after the loop" code on its own. See
    # test_parse_decision_always_answers_on_the_last_attempt below, which
    # guards that assumption directly against the real (unmocked) function.
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

        await _ask(deps, "a hard question")

    gaps = list_knowledge_gaps(open_store(deps.policy.db_path))
    assert len(gaps) == 1
    assert gaps[0].reason == "repeated_query_no_answer"


def test_parse_decision_always_answers_on_the_last_attempt() -> None:
    # Drives the real (unmocked) _parse_decision directly -- guards the
    # exact assumption the two tests above depend on, found by review: with
    # remaining_searches == 0, even a response that looks like a search
    # request must come back as AnswerDecision, never SearchDecision. If
    # this ever stops being true, the "repeated query" gap-logging path
    # would silently become reachable a different way and the test above
    # would need rewriting.
    from library_mcp.keeper_model import _parse_decision

    decision = _parse_decision("SEARCH: one more thing to check", remaining_searches=0)
    assert isinstance(decision, AnswerDecision)
