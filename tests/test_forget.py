from pathlib import Path
from unittest.mock import AsyncMock, patch

from library_mcp.audit import AuditLog
from library_mcp.config import ParsePolicy
from library_mcp.servers.parse_server import _Deps, _forget, _learn_text
from library_mcp.store import list_books, open_store

from conftest import run_and_wait


def _deps(tmp_path: Path) -> _Deps:
    policy = ParsePolicy(
        ollama_base_url="http://unused:11434",
        inbox_path=tmp_path / "inbox",
        db_path=tmp_path / "knowledge.db",
    )
    return _Deps(policy=policy, audit=AuditLog(tmp_path / "audit.jsonl"))


async def _learn_one(deps: _Deps, title: str, text: str = "some content") -> None:
    with patch("library_mcp.servers.parse_server.EmbeddingClient") as mock_client:
        mock_client.return_value.embed = AsyncMock(return_value=[1.0, 0.0])
        await run_and_wait(_learn_text, deps, title, text, f"https://example.com/{title}")


async def test_forget_removes_an_exact_title_match(tmp_path: Path) -> None:
    deps = _deps(tmp_path)
    await _learn_one(deps, "Networking With Python")

    result = _forget(deps, "Networking With Python")

    assert "Forgot 'Networking With Python'" in result
    conn = open_store(deps.policy.db_path)
    assert list_books(conn) == []


async def test_forget_matches_a_partial_case_insensitive_title(tmp_path: Path) -> None:
    deps = _deps(tmp_path)
    await _learn_one(deps, "Networking With Python")

    result = _forget(deps, "networking")

    assert "Forgot 'Networking With Python'" in result
    conn = open_store(deps.policy.db_path)
    assert list_books(conn) == []


async def test_forget_refuses_an_ambiguous_partial_match(tmp_path: Path) -> None:
    deps = _deps(tmp_path)
    await _learn_one(deps, "Python Fundamentals")
    await _learn_one(deps, "Python Penetration Testing", text="other content")

    result = _forget(deps, "python")

    assert "more than one book" in result
    conn = open_store(deps.policy.db_path)
    assert len(list_books(conn)) == 2  # neither deleted


def test_forget_refuses_no_match(tmp_path: Path) -> None:
    deps = _deps(tmp_path)
    result = _forget(deps, "Nonexistent Book")
    assert "No learned book matches" in result


def test_forget_refuses_empty_title(tmp_path: Path) -> None:
    deps = _deps(tmp_path)
    result = _forget(deps, "   ")
    assert "Refused" in result
