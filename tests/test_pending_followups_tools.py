"""Tests for the two new keeper_server MCP tools backing the
knowledge_followup gateway plugin (docs/planning/knowledge-gap-research.md
§5.1, §9 build step 7) -- unit-testable directly against a seeded test
knowledge.db, independent of the plugin or a live turn."""

from pathlib import Path

from library_mcp.audit import AuditLog, Event
from library_mcp.config import KeeperPolicy
from library_mcp.servers.keeper_server import _Deps, build_server
from library_mcp.store import (
    create_pending_followup,
    mark_gap_resolved,
    open_store,
    record_knowledge_gap,
)


def _deps(tmp_path: Path) -> _Deps:
    policy = KeeperPolicy(ollama_base_url="http://unused:11434", db_path=tmp_path / "knowledge.db")
    return _Deps(policy=policy, audit=AuditLog(tmp_path / "audit.jsonl"))


def _seed_followup(deps: _Deps, chat_id: str = "chat1") -> tuple[int, int]:
    conn = open_store(deps.policy.db_path)
    record_knowledge_gap(
        conn, "who taught Marcus Aurelius philosophy", "no_matches", "2026-01-01T00:00:00+00:00"
    )
    gap_id = conn.execute("SELECT id FROM knowledge_gaps").fetchone()[0]
    mark_gap_resolved(conn, gap_id)
    followup_id = create_pending_followup(
        conn,
        gap_id,
        chat_id,
        "who taught Marcus Aurelius philosophy",
        "external_research",
        "Wikipedia has an answer",
        "2026-01-02T00:00:00+00:00",
    )
    return gap_id, followup_id


def _find_tool(app, name: str):
    # FastMCP registers tools on an internal manager; the simplest
    # test-stable way to reach the underlying function directly (matching
    # this file's own goal: unit-test the tool logic, not the MCP
    # transport) is to call build_server and pull the tool's fn attribute.
    tool = app._tool_manager._tools[name]  # type: ignore[attr-defined]
    return tool.fn


def test_list_pending_followups_returns_seeded_row(tmp_path: Path) -> None:
    deps = _deps(tmp_path)
    _gap_id, followup_id = _seed_followup(deps)
    app = build_server(deps.policy, deps.audit)
    list_fn = _find_tool(app, "list_pending_followups")

    result = list_fn(chat_id="chat1")

    assert str(followup_id) in result
    assert "who taught Marcus Aurelius philosophy" in result
    assert "external_research" in result


def test_list_pending_followups_empty_for_unknown_chat(tmp_path: Path) -> None:
    deps = _deps(tmp_path)
    _seed_followup(deps, chat_id="chat1")
    app = build_server(deps.policy, deps.audit)
    list_fn = _find_tool(app, "list_pending_followups")

    assert list_fn(chat_id="someone_else") == "[]"


def test_mark_followup_delivered_writes_the_audit_event_and_db_state(tmp_path: Path) -> None:
    deps = _deps(tmp_path)
    gap_id, followup_id = _seed_followup(deps)
    app = build_server(deps.policy, deps.audit)
    mark_fn = _find_tool(app, "mark_followup_delivered")
    list_fn = _find_tool(app, "list_pending_followups")

    result = mark_fn(followup_ids=[followup_id])

    assert str(followup_id) in result
    # No longer pending once delivered.
    assert list_fn(chat_id="chat1") == "[]"
    # followup_injected audit event was written server-side (§5.1/§8.2).
    audit_text = (tmp_path / "audit.jsonl").read_text(encoding="utf-8")
    assert Event.FOLLOWUP_INJECTED in audit_text
    assert str(gap_id) in audit_text


def test_mark_followup_delivered_is_idempotent(tmp_path: Path) -> None:
    deps = _deps(tmp_path)
    _gap_id, followup_id = _seed_followup(deps)
    app = build_server(deps.policy, deps.audit)
    mark_fn = _find_tool(app, "mark_followup_delivered")

    first = mark_fn(followup_ids=[followup_id])
    second = mark_fn(followup_ids=[followup_id])

    assert str(followup_id) in first
    import json

    assert json.loads(second)["delivered"] == []


def test_pending_followups_tools_are_excluded_by_config_convention(tmp_path: Path) -> None:
    # This test documents the contract rather than exercising config.yaml
    # directly (that file lives on the host, outside this repo) -- see
    # docs/architecture/knowledge-gap-research.md for the actual deployed
    # config.yaml `tools.exclude` entry this depends on.
    deps = _deps(tmp_path)
    app = build_server(deps.policy, deps.audit)
    assert "list_pending_followups" in app._tool_manager._tools  # type: ignore[attr-defined]
    assert "mark_followup_delivered" in app._tool_manager._tools  # type: ignore[attr-defined]
