from pathlib import Path
from unittest.mock import AsyncMock, patch

from library_mcp.audit import AuditLog
from library_mcp.config import ParsePolicy
from library_mcp.servers.parse_server import _Deps, _learn_text
from library_mcp.store import chunk_count, list_books, open_store

from conftest import run_and_wait


def _deps(tmp_path: Path) -> _Deps:
    policy = ParsePolicy(
        ollama_base_url="http://unused:11434",
        inbox_path=tmp_path / "inbox",
        db_path=tmp_path / "knowledge.db",
    )
    return _Deps(policy=policy, audit=AuditLog(tmp_path / "audit.jsonl"))


async def test_learn_text_ingests_fetched_content(tmp_path: Path) -> None:
    deps = _deps(tmp_path)
    with patch("library_mcp.servers.parse_server.EmbeddingClient") as mock_client:
        mock_client.return_value.embed = AsyncMock(return_value=[1.0, 0.0])
        result = await run_and_wait(
            _learn_text, deps, "Some Article", "the fetched page content", "https://example.com/a"
        )

    assert "Started learning" in result
    assert "Some Article" in result
    conn = open_store(deps.policy.db_path)
    assert "Some Article" in list_books(conn)
    assert chunk_count(conn) == 1


async def test_learn_text_refuses_duplicate_content(tmp_path: Path) -> None:
    deps = _deps(tmp_path)
    with patch("library_mcp.servers.parse_server.EmbeddingClient") as mock_client:
        mock_client.return_value.embed = AsyncMock(return_value=[1.0, 0.0])
        await run_and_wait(_learn_text, deps, "Some Article", "the fetched page content", "https://example.com/a")
        result = _learn_text(deps, "Some Article Again", "the fetched page content", "https://example.com/a")

    assert "already have this content" in result
    conn = open_store(deps.policy.db_path)
    assert len(list_books(conn)) == 1  # not duplicated


def test_learn_text_refuses_empty_text(tmp_path: Path) -> None:
    deps = _deps(tmp_path)
    result = _learn_text(deps, "Empty", "   ", "https://example.com/a")
    assert "Refused" in result


def test_learn_text_refuses_empty_title(tmp_path: Path) -> None:
    deps = _deps(tmp_path)
    result = _learn_text(deps, "  ", "some text", "https://example.com/a")
    assert "Refused" in result
