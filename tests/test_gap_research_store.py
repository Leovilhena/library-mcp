"""Schema migration + storage-layer tests for knowledge-gap research
(docs/planning/knowledge-gap-research.md §4, §9 build step 0)."""

from pathlib import Path

from library_mcp.store import (
    create_pending_followup,
    expire_stale_followups,
    increment_external_attempt_count,
    latest_external_research,
    list_open_gaps,
    list_pending_followups,
    mark_followups_delivered,
    mark_gap_resolved,
    open_store,
    record_external_research,
    record_knowledge_gap,
    set_giving_up,
)


def test_open_store_creates_the_new_tables_and_columns(tmp_path: Path) -> None:
    conn = open_store(tmp_path / "test.db")
    # Additive migration didn't error, and the new columns exist with sane
    # defaults for a freshly-created gap.
    record_knowledge_gap(conn, "a question", "no_matches", "2026-01-01T00:00:00+00:00")
    gaps = list_open_gaps(conn)
    assert len(gaps) == 1
    assert gaps[0].external_attempt_count == 0
    assert gaps[0].giving_up is False


def test_open_store_migrates_a_pre_gap_research_database(tmp_path: Path) -> None:
    # Simulates the real knowledge.db as it existed before this feature:
    # knowledge_gaps exists but without external_attempt_count/giving_up,
    # and no external_research/pending_followups tables at all.
    import sqlite3

    db_path = tmp_path / "old.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE books (
            id INTEGER PRIMARY KEY, title TEXT NOT NULL, source_path TEXT NOT NULL,
            ingested_at TEXT NOT NULL
        );
        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY, book_id INTEGER NOT NULL, chunk_index INTEGER NOT NULL,
            section TEXT, text TEXT NOT NULL, embedding TEXT NOT NULL
        );
        CREATE TABLE knowledge_gaps (
            id INTEGER PRIMARY KEY, question TEXT NOT NULL, reason TEXT NOT NULL,
            first_asked_at TEXT NOT NULL, last_asked_at TEXT NOT NULL,
            times_asked INTEGER NOT NULL DEFAULT 1, resolved INTEGER NOT NULL DEFAULT 0
        );
        INSERT INTO knowledge_gaps (question, reason, first_asked_at, last_asked_at)
        VALUES ('pre-existing gap', 'no_matches', '2026-01-01', '2026-01-01');
        """
    )
    conn.commit()
    conn.close()

    conn = open_store(db_path)
    gaps = list_open_gaps(conn)
    assert len(gaps) == 1
    assert gaps[0].question == "pre-existing gap"
    assert gaps[0].external_attempt_count == 0
    assert gaps[0].giving_up is False
    # New tables are queryable without error.
    assert list_pending_followups(conn, "chat1") == []


def test_external_research_retry_cap_bookkeeping(tmp_path: Path) -> None:
    conn = open_store(tmp_path / "test.db")
    record_knowledge_gap(conn, "a question", "no_matches", "2026-01-01T00:00:00+00:00")
    gap_id = list_open_gaps(conn)[0].id

    record_external_research(
        conn, gap_id, "a question", "wikipedia", "content_miss", "2026-01-02T00:00:00+00:00"
    )
    count = increment_external_attempt_count(conn, gap_id)
    assert count == 1
    assert list_open_gaps(conn)[0].giving_up is False

    increment_external_attempt_count(conn, gap_id)
    count = increment_external_attempt_count(conn, gap_id)
    assert count == 3
    # Caller decides when to actually set giving_up -- this function only
    # bumps the counter, mirroring gap_research.py's own separation.
    set_giving_up(conn, gap_id)
    assert list_open_gaps(conn)[0].giving_up is True


def test_infra_down_never_increments_the_attempt_counter(tmp_path: Path) -> None:
    conn = open_store(tmp_path / "test.db")
    record_knowledge_gap(conn, "q", "no_matches", "2026-01-01T00:00:00+00:00")
    gap_id = list_open_gaps(conn)[0].id

    record_external_research(
        conn, gap_id, "q", "wikipedia", "infra_down", "2026-01-02T00:00:00+00:00"
    )
    # No increment_external_attempt_count call for infra_down -- verifying
    # the store layer doesn't do it implicitly either.
    assert list_open_gaps(conn)[0].external_attempt_count == 0


def test_latest_external_research_returns_the_most_recent_row(tmp_path: Path) -> None:
    conn = open_store(tmp_path / "test.db")
    record_knowledge_gap(conn, "q", "no_matches", "2026-01-01T00:00:00+00:00")
    gap_id = list_open_gaps(conn)[0].id

    record_external_research(
        conn, gap_id, "q", "wikipedia", "content_miss", "2026-01-01T00:00:00+00:00"
    )
    record_external_research(conn, gap_id, "q", "wikipedia", "found", "2026-01-08T00:00:00+00:00")

    latest = latest_external_research(conn, gap_id, "wikipedia")
    assert latest is not None
    assert latest.outcome == "found"
    assert latest.researched_at == "2026-01-08T00:00:00+00:00"

    assert latest_external_research(conn, gap_id, "wikiquote") is None


def test_pending_followups_lifecycle(tmp_path: Path) -> None:
    conn = open_store(tmp_path / "test.db")
    record_knowledge_gap(conn, "q", "no_matches", "2026-01-01T00:00:00+00:00")
    gap_id = list_open_gaps(conn)[0].id
    mark_gap_resolved(conn, gap_id)

    followup_id = create_pending_followup(
        conn, gap_id, "chat1", "q", "book_deepen", "found it", "2026-01-01T00:00:00+00:00"
    )

    # Not visible to a different chat.
    assert list_pending_followups(conn, "chat2") == []
    rows = list_pending_followups(conn, "chat1")
    assert len(rows) == 1
    assert rows[0].id == followup_id
    assert rows[0].resolution_kind == "book_deepen"

    delivered = mark_followups_delivered(conn, [followup_id], "2026-01-02T00:00:00+00:00")
    assert delivered == [followup_id]
    # Once delivered, it no longer shows up as pending.
    assert list_pending_followups(conn, "chat1") == []

    # Idempotent: marking again is a no-op, not an error.
    assert mark_followups_delivered(conn, [followup_id], "2026-01-03T00:00:00+00:00") == []


def test_pending_followups_respects_the_per_chat_limit(tmp_path: Path) -> None:
    conn = open_store(tmp_path / "test.db")
    for i in range(3):
        record_knowledge_gap(conn, f"q{i}", "no_matches", "2026-01-01T00:00:00+00:00")
    gap_ids = [g.id for g in list_open_gaps(conn)]
    for i, gap_id in enumerate(gap_ids):
        mark_gap_resolved(conn, gap_id)
        create_pending_followup(
            conn,
            gap_id,
            "chat1",
            f"q{i}",
            "book_deepen",
            f"answer {i}",
            f"2026-01-0{i + 1}T00:00:00+00:00",
        )

    rows = list_pending_followups(conn, "chat1", limit=2)
    assert len(rows) == 2
    # Oldest-resolved-first.
    assert rows[0].resolved_at < rows[1].resolved_at


def test_expire_stale_followups(tmp_path: Path) -> None:
    conn = open_store(tmp_path / "test.db")
    record_knowledge_gap(conn, "old q", "no_matches", "2026-01-01T00:00:00+00:00")
    record_knowledge_gap(conn, "new q", "no_matches", "2026-01-01T00:00:00+00:00")
    old_gap, new_gap = (g.id for g in list_open_gaps(conn))
    mark_gap_resolved(conn, old_gap)
    mark_gap_resolved(conn, new_gap)
    create_pending_followup(
        conn, old_gap, "chat1", "old q", "book_deepen", "old answer", "2026-01-01T00:00:00+00:00"
    )
    create_pending_followup(
        conn, new_gap, "chat1", "new q", "book_deepen", "new answer", "2026-01-20T00:00:00+00:00"
    )

    expired = expire_stale_followups(conn, "2026-01-25T00:00:00+00:00", "2026-01-10T00:00:00+00:00")

    assert len(expired) == 1
    assert expired[0].gap_id == old_gap
    # The still-fresh one is untouched and still surfaces normally.
    remaining = list_pending_followups(conn, "chat1")
    assert len(remaining) == 1
    assert remaining[0].gap_id == new_gap
