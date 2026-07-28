from pathlib import Path
from unittest.mock import patch

from library_mcp.audit import AuditLog
from library_mcp.config import ParsePolicy
from library_mcp.servers.parse_server import _Deps, _learn


def _deps(tmp_path: Path, documents_path: Path | None) -> _Deps:
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    policy = ParsePolicy(
        ollama_base_url="http://unused:11434",
        inbox_path=inbox,
        documents_path=documents_path,
        db_path=tmp_path / "knowledge.db",
    )
    return _Deps(policy=policy, audit=AuditLog(tmp_path / "audit.jsonl"))


def test_learn_finds_file_in_telegram_document_cache_not_just_inbox(tmp_path: Path) -> None:
    # The exact real bug: a PDF sent via Telegram lands in Hermes' generic
    # attachment cache, not the dedicated inbox. learn() must still find it.
    documents = tmp_path / "documents"
    documents.mkdir()
    fixture = Path(__file__).parent / "fixtures" / "networking-with-python.pdf"
    (documents / "networking-with-python.pdf").write_bytes(fixture.read_bytes())

    deps = _deps(tmp_path, documents_path=documents)
    with patch("library_mcp.servers.parse_server.EmbeddingClient") as mock_client:
        mock_client.return_value.embed.return_value = [1.0, 0.0]
        result = _learn(deps, "networking-with-python.pdf")

    assert "Learned" in result


def test_learn_prefers_inbox_over_documents_when_both_have_it(tmp_path: Path) -> None:
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "same-name.pdf").write_bytes(b"should not be used")
    fixture = Path(__file__).parent / "fixtures" / "networking-with-python.pdf"
    deps = _deps(tmp_path, documents_path=documents)
    (deps.policy.inbox_path / "same-name.pdf").write_bytes(fixture.read_bytes())

    with patch("library_mcp.servers.parse_server.EmbeddingClient") as mock_client:
        mock_client.return_value.embed.return_value = [1.0, 0.0]
        result = _learn(deps, "same-name.pdf")

    assert "Learned" in result  # the real PDF parsed cleanly, not the bogus bytes


def test_learn_refuses_when_not_in_either_root(tmp_path: Path) -> None:
    deps = _deps(tmp_path, documents_path=tmp_path / "documents")
    result = _learn(deps, "nonexistent.pdf")
    assert "Refused" in result
    assert "not found in the inbox or recent documents" in result


def test_learn_works_without_documents_path_configured(tmp_path: Path) -> None:
    deps = _deps(tmp_path, documents_path=None)
    result = _learn(deps, "nonexistent.pdf")
    assert "Refused" in result
