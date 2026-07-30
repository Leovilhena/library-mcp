"""Fix 1 (2026-07-30 zero-tool-call fabrication incident, session
20260729_160958_da2b002f): the `ask_library` tool description previously
read "...only call it again in the same turn if the user asked about a
genuinely different topic", worded broadly enough that a small,
~40-60%-tool-calling-reliable model could read "don't call it again" as
spanning across turns, not just within one -- and on 2026-07-30 that is
exactly what happened: the model asserted a prior turn's (yesterday's, real)
retrieval struggle as still-current fact without calling the tool at all
today.

This is a description-only change (no schema/behavior implications), so the
only thing worth asserting is that the new wording is actually present and
unambiguous about per-question, not per-turn, scope.
"""

from pathlib import Path

from library_mcp.audit import AuditLog
from library_mcp.config import KeeperPolicy
from library_mcp.servers.keeper_server import build_server


def _ask_library_description(tmp_path: Path) -> str:
    policy = KeeperPolicy(ollama_base_url="http://unused:11434", db_path=tmp_path / "knowledge.db")
    server = build_server(policy, AuditLog(tmp_path / "audit.jsonl"))
    tools = server._tool_manager._tools  # FastMCP internal registry
    return tools["ask_library"].description or ""


def test_ask_library_description_scopes_reuse_to_the_same_turn(tmp_path: Path) -> None:
    desc = _ask_library_description(tmp_path)
    lowered = desc.lower()
    # Still discourages the same-turn double-call this description was built
    # to prevent in the first place.
    assert "call it once per question" in lowered
    # New: makes cross-turn reuse of a stale outcome explicit and wrong.
    assert "new question" in lowered
    assert "prior attempt's outcome" in lowered or "prior attempt" in lowered
    assert "never assume" in lowered
    # The old ambiguous phrasing ("only call it again in the same turn if")
    # must be gone -- it's what let a small model over-generalize "already
    # tried this" across turns instead of within one.
    assert "only call it again in the same turn if" not in lowered
