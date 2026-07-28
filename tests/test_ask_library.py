from pathlib import Path
from unittest.mock import AsyncMock, patch

from library_mcp.audit import AuditLog
from library_mcp.config import KeeperPolicy
from library_mcp.keeper_model import AnswerDecision, SearchDecision
from library_mcp.servers.keeper_server import _Deps, _ask
from library_mcp.store import add_book, add_chunk, commit, list_knowledge_gaps, open_store


def _deps(tmp_path: Path) -> _Deps:
    policy = KeeperPolicy(ollama_base_url="http://unused:11434", db_path=tmp_path / "knowledge.db")
    return _Deps(policy=policy, audit=AuditLog(tmp_path / "audit.jsonl"))


async def test_ask_library_logs_a_gap_when_nothing_matches(tmp_path: Path) -> None:
    deps = _deps(tmp_path)
    with (
        patch("library_mcp.servers.keeper_server.EmbeddingClient") as mock_embed,
        patch("library_mcp.servers.keeper_server.ReasoningClient") as mock_reason,
    ):
        mock_embed.return_value.embed = AsyncMock(return_value=[1.0, 0.0])
        mock_reason.return_value.decide = AsyncMock(return_value=SearchDecision(query="rephrased"))

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
        mock_reason.return_value.decide = AsyncMock(return_value=AnswerDecision(text="Here's the answer."))

        result = await _ask(deps, "a real question")

    assert result == "Here's the answer."
    assert list_knowledge_gaps(open_store(deps.policy.db_path)) == []


async def test_ask_library_logs_max_searches_exhausted_reason(tmp_path: Path) -> None:
    # Real matches exist, but the reasoner never settles on an answer --
    # a different gap reason than pure "no_matches".
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
        mock_reason.return_value.decide = AsyncMock(return_value=SearchDecision(query="still looking"))

        await _ask(deps, "a hard question")

    gaps = list_knowledge_gaps(open_store(deps.policy.db_path))
    assert len(gaps) == 1
    assert gaps[0].reason == "max_searches_exhausted"
