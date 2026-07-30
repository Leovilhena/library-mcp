"""Tests for the book-deepen step (docs/planning/knowledge-gap-research.md
§2, §9 build step 1) -- mirrors test_ask_library.py's mocking style."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

from library_mcp.audit import AuditLog, Event
from library_mcp.book_deepen import DeepenConfig, deepen
from library_mcp.keeper_model import AnswerDecision, SearchDecision
from library_mcp.store import add_book, add_chunk, commit, open_store


def _config() -> DeepenConfig:
    return DeepenConfig(ollama_base_url="http://unused:11434", top_k=20)


async def test_deepen_returns_the_answer_when_results_back_it(tmp_path: Path) -> None:
    conn = open_store(tmp_path / "test.db")
    book_id = add_book(conn, "Meditations", "/inbox/med.pdf", "2026-01-01")
    add_chunk(conn, book_id, 0, "Book 2", "on the fear of death", [1.0, 0.0])
    commit(conn)
    audit_path = tmp_path / "audit.jsonl"
    audit = AuditLog(audit_path)

    with (
        patch("library_mcp.book_deepen.EmbeddingClient") as mock_embed,
        patch("library_mcp.book_deepen.ReasoningClient") as mock_reason,
    ):
        mock_embed.return_value.embed = AsyncMock(return_value=[1.0, 0.0])
        mock_reason.return_value.decide = AsyncMock(
            return_value=AnswerDecision(text="Book 2 discusses accepting mortality.")
        )

        answer = await deepen(conn, audit, gap_id=1, question="fear of death?", config=_config())

    assert answer == "Book 2 discusses accepting mortality."
    events = audit_path.read_text(encoding="utf-8")
    assert Event.BOOK_DEEPEN_ATTEMPTED in events
    assert Event.BOOK_DEEPEN_RESOLVED in events


async def test_deepen_returns_none_when_no_results_back_the_answer(tmp_path: Path) -> None:
    # Same "AnswerDecision with zero backing results = not a real answer"
    # rule _ask() itself follows (§0) -- the deepen pass must not report a
    # confident resolution when the search genuinely found nothing.
    conn = open_store(tmp_path / "test.db")
    audit_path = tmp_path / "audit.jsonl"
    audit = AuditLog(audit_path)

    with (
        patch("library_mcp.book_deepen.EmbeddingClient") as mock_embed,
        patch("library_mcp.book_deepen.ReasoningClient") as mock_reason,
    ):
        mock_embed.return_value.embed = AsyncMock(return_value=[1.0, 0.0])
        mock_reason.return_value.decide = AsyncMock(
            return_value=AnswerDecision(text="Nothing on this in the passages.")
        )

        answer = await deepen(conn, audit, gap_id=1, question="obscure question", config=_config())

    assert answer is None
    events = audit_path.read_text(encoding="utf-8")
    assert Event.BOOK_DEEPEN_NO_ANSWER in events


async def test_deepen_stops_on_a_repeated_query_without_crashing(tmp_path: Path) -> None:
    conn = open_store(tmp_path / "test.db")
    book_id = add_book(conn, "Book", "/inbox/b.pdf", "2026-01-01")
    add_chunk(conn, book_id, 0, "p1", "somewhat related", [1.0, 0.0])
    commit(conn)
    audit_path = tmp_path / "audit.jsonl"
    audit = AuditLog(audit_path)

    with (
        patch("library_mcp.book_deepen.EmbeddingClient") as mock_embed,
        patch("library_mcp.book_deepen.ReasoningClient") as mock_reason,
    ):
        mock_embed.return_value.embed = AsyncMock(return_value=[1.0, 0.0])
        mock_reason.return_value.decide = AsyncMock(return_value=SearchDecision(query="a hard question"))

        answer = await deepen(conn, audit, gap_id=1, question="a hard question", config=_config())

    assert answer is None


async def test_deepen_max_searches_is_still_enforced(tmp_path: Path) -> None:
    # §2: widening top_k/upgrading the model doesn't remove the loop's
    # ceiling -- an unbounded loop on an overnight cron job is still the
    # "stuck agent" failure max_searches exists to prevent.
    conn = open_store(tmp_path / "test.db")
    book_id = add_book(conn, "Book", "/inbox/b.pdf", "2026-01-01")
    add_chunk(conn, book_id, 0, "p1", "text", [1.0, 0.0])
    commit(conn)
    audit_path = tmp_path / "audit.jsonl"
    audit = AuditLog(audit_path)

    call_count = 0

    async def _always_search(query, context, remaining):
        nonlocal call_count
        call_count += 1
        return SearchDecision(query=f"query {call_count}")

    with (
        patch("library_mcp.book_deepen.EmbeddingClient") as mock_embed,
        patch("library_mcp.book_deepen.ReasoningClient") as mock_reason,
    ):
        mock_embed.return_value.embed = AsyncMock(return_value=[1.0, 0.0])
        mock_reason.return_value.decide = AsyncMock(side_effect=_always_search)

        config = DeepenConfig(ollama_base_url="http://unused:11434", top_k=20, max_searches=3)
        answer = await deepen(conn, audit, gap_id=1, question="q0", config=config)

    assert answer is None
    assert call_count == 3
