"""Tests for the new keeper_server MCP tool backing the gateway's
sandbox_redirect plugin (docs/incidents/2026-07-30-sandbox-vs-ask-library.md)
-- unit-testable directly against a seeded test knowledge.db, independent of
the plugin or a live turn. Matches the style of test_pending_followups_tools.py,
built the same day for the same hidden-tool pattern."""

import json
from pathlib import Path

from library_mcp.audit import AuditLog, Event
from library_mcp.config import KeeperPolicy
from library_mcp.servers.keeper_server import _Deps, build_server
from library_mcp.store import add_book, open_store


def _deps(tmp_path: Path) -> _Deps:
    policy = KeeperPolicy(ollama_base_url="http://unused:11434", db_path=tmp_path / "knowledge.db")
    return _Deps(policy=policy, audit=AuditLog(tmp_path / "audit.jsonl"))


def _find_tool(app, name: str):
    tool = app._tool_manager._tools[name]  # type: ignore[attr-defined]
    return tool.fn


def test_find_book_by_filename_returns_match_for_done_book(tmp_path: Path) -> None:
    deps = _deps(tmp_path)
    conn = open_store(deps.policy.db_path)
    add_book(
        conn,
        "Awakening the Heroes Within",
        "/documents/doc_8d70205d2619_Carol_S_Pearson_Awakening_the_Heroes_Within.epub",
        "2026-07-28",
        status="done",
    )
    app = build_server(deps.policy, deps.audit)
    lookup_fn = _find_tool(app, "find_book_by_filename")

    result = json.loads(
        lookup_fn(
            filename="/opt/data/cache/documents/doc_8d70205d2619_Carol_S_Pearson_Awakening_the_Heroes_Within.epub"
        )
    )

    assert result["found"] is True
    assert result["title"] == "Awakening the Heroes Within"
    assert result["status"] == "done"


def test_find_book_by_filename_no_match_returns_found_false(tmp_path: Path) -> None:
    deps = _deps(tmp_path)
    conn = open_store(deps.policy.db_path)
    add_book(conn, "Some Book", "/documents/a.pdf", "2026-01-01", status="done")
    app = build_server(deps.policy, deps.audit)
    lookup_fn = _find_tool(app, "find_book_by_filename")

    result = json.loads(lookup_fn(filename="unrelated_file.pdf"))

    assert result == {"found": False}


def test_find_book_by_filename_writes_audit_event(tmp_path: Path) -> None:
    deps = _deps(tmp_path)
    conn = open_store(deps.policy.db_path)
    add_book(conn, "Some Book", "/documents/a.pdf", "2026-01-01", status="done")
    app = build_server(deps.policy, deps.audit)
    lookup_fn = _find_tool(app, "find_book_by_filename")

    lookup_fn(filename="a.pdf")

    audit_text = (tmp_path / "audit.jsonl").read_text(encoding="utf-8")
    assert Event.FILENAME_LOOKED_UP in audit_text
    assert '"found": true' in audit_text


def test_find_book_by_filename_tool_is_excluded_by_config_convention(tmp_path: Path) -> None:
    # This documents the contract rather than exercising config.yaml
    # directly (that file lives on the host, outside this repo) -- see
    # docs/architecture/sandbox-mcp.md for the actual deployed
    # `tools.exclude` entry this depends on.
    deps = _deps(tmp_path)
    app = build_server(deps.policy, deps.audit)
    assert "find_book_by_filename" in app._tool_manager._tools  # type: ignore[attr-defined]
