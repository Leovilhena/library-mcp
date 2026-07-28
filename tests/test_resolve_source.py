from pathlib import Path
from unittest.mock import AsyncMock, patch

from conftest import run_and_wait
from library_mcp.audit import AuditLog
from library_mcp.config import ParsePolicy
from library_mcp.servers.parse_server import _Deps, _learn
from library_mcp.store import list_books, open_store


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


async def test_learn_finds_file_in_telegram_document_cache_not_just_inbox(tmp_path: Path) -> None:
    # The exact real bug: a PDF sent via Telegram lands in Hermes' generic
    # attachment cache, not the dedicated inbox. learn() must still find it.
    documents = tmp_path / "documents"
    documents.mkdir()
    fixture = Path(__file__).parent / "fixtures" / "networking-with-python.pdf"
    (documents / "networking-with-python.pdf").write_bytes(fixture.read_bytes())

    deps = _deps(tmp_path, documents_path=documents)
    with patch("library_mcp.servers.parse_server.EmbeddingClient") as mock_client:
        mock_client.return_value.embed = AsyncMock(return_value=[1.0, 0.0])
        result = await run_and_wait(_learn, deps, "networking-with-python.pdf")

    assert "Started learning" in result
    conn = open_store(deps.policy.db_path)
    assert list_books(conn)  # actually finished ingesting, not just accepted


async def test_learn_prefers_inbox_over_documents_when_both_have_it(tmp_path: Path) -> None:
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "same-name.pdf").write_bytes(b"should not be used")
    fixture = Path(__file__).parent / "fixtures" / "networking-with-python.pdf"
    deps = _deps(tmp_path, documents_path=documents)
    (deps.policy.inbox_path / "same-name.pdf").write_bytes(fixture.read_bytes())

    with patch("library_mcp.servers.parse_server.EmbeddingClient") as mock_client:
        mock_client.return_value.embed = AsyncMock(return_value=[1.0, 0.0])
        result = await run_and_wait(_learn, deps, "same-name.pdf")

    assert "Started learning" in result  # the real PDF parsed cleanly, not the bogus bytes


def test_learn_refuses_when_not_in_either_root(tmp_path: Path) -> None:
    deps = _deps(tmp_path, documents_path=tmp_path / "documents")
    result = _learn(deps, "nonexistent.pdf")
    assert "Refused" in result
    assert "not found in the inbox or recent documents" in result


def test_learn_works_without_documents_path_configured(tmp_path: Path) -> None:
    deps = _deps(tmp_path, documents_path=None)
    result = _learn(deps, "nonexistent.pdf")
    assert "Refused" in result


async def test_learn_refuses_duplicate_file(tmp_path: Path) -> None:
    documents = tmp_path / "documents"
    documents.mkdir()
    fixture = Path(__file__).parent / "fixtures" / "networking-with-python.pdf"
    (documents / "networking-with-python.pdf").write_bytes(fixture.read_bytes())
    # A re-upload of the identical file, e.g. Hermes' cache assigning it a
    # new doc_<hash>_ prefix on a second Telegram send -- different name,
    # identical bytes.
    (documents / "networking-with-python-copy.pdf").write_bytes(fixture.read_bytes())

    deps = _deps(tmp_path, documents_path=documents)
    with patch("library_mcp.servers.parse_server.EmbeddingClient") as mock_client:
        mock_client.return_value.embed = AsyncMock(return_value=[1.0, 0.0])
        await run_and_wait(_learn, deps, "networking-with-python.pdf")
        result = _learn(deps, "networking-with-python-copy.pdf")

    assert "already have this book" in result
    conn = open_store(deps.policy.db_path)
    assert len(list_books(conn)) == 1


async def test_learn_fuzzy_matches_a_dropped_doc_prefix(tmp_path: Path) -> None:
    # The exact real failure, 2026-07-28: given "[The user sent a document:
    # ...]" with the real name "doc_60802e8476de__Kozai,_Toyoki_Niu.pdf",
    # the model instead called learn with a reconstructed-from-memory name
    # missing the doc_<hash>_ prefix and using different punctuation. Three
    # separate real attempts all failed this same way before this fix.
    documents = tmp_path / "documents"
    documents.mkdir()
    real_name = "doc_60802e8476de__Kozai,_Toyoki_Niu,_Genhua_Takagaki,_Michiko_Plant_factory.pdf"
    fixture = Path(__file__).parent / "fixtures" / "networking-with-python.pdf"
    (documents / real_name).write_bytes(fixture.read_bytes())

    deps = _deps(tmp_path, documents_path=documents)
    reconstructed_name = "_Kozai___Toyoki_Niu___Genhua_Takagaki_Michiko_Plant_factory.pdf"
    with patch("library_mcp.servers.parse_server.EmbeddingClient") as mock_client:
        mock_client.return_value.embed = AsyncMock(return_value=[1.0, 0.0])
        result = await run_and_wait(_learn, deps, reconstructed_name)

    assert "Started learning" in result
    conn = open_store(deps.policy.db_path)
    assert list_books(conn)  # actually resolved and ingested, not refused


def test_learn_fuzzy_match_refuses_when_ambiguous(tmp_path: Path) -> None:
    # Two different real files that would normalize to the same fuzzy key --
    # must refuse rather than silently guess which one was meant.
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "doc_aaa111_Some Book.pdf").write_bytes(b"one")
    (documents / "doc_bbb222_Some_Book.pdf").write_bytes(b"two")

    deps = _deps(tmp_path, documents_path=documents)
    result = _learn(deps, "Some_Book.pdf")

    assert "Refused" in result


async def test_learn_exact_match_is_never_shadowed_by_a_fuzzy_one(tmp_path: Path) -> None:
    # If the exact given name exists, it must win even if a differently
    # named file would also fuzzy-match -- exact intent beats a guess.
    documents = tmp_path / "documents"
    documents.mkdir()
    fixture = Path(__file__).parent / "fixtures" / "networking-with-python.pdf"
    (documents / "exact-name.pdf").write_bytes(fixture.read_bytes())
    (documents / "doc_xyz_exact-name.pdf").write_bytes(b"decoy, would also fuzzy-match")

    deps = _deps(tmp_path, documents_path=documents)
    with patch("library_mcp.servers.parse_server.EmbeddingClient") as mock_client:
        mock_client.return_value.embed = AsyncMock(return_value=[1.0, 0.0])
        result = await run_and_wait(_learn, deps, "exact-name.pdf")

    assert "Started learning" in result
    conn = open_store(deps.policy.db_path)
    assert list_books(conn)
