"""Append-only JSONL audit trail.

Same shape as sandbox-mcp's audit.py deliberately -- the whole project's
"judge by the log, not the model's word" principle only works if every
component logs the same way. Kept as a small standalone copy rather than a
shared dependency between the two repos, matching sandbox-mcp's own
self-contained-repo distribution model.
"""

from __future__ import annotations

import json
import stat
import sys
import threading
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any


class Event(StrEnum):
    INGESTED = "ingested"
    INGEST_DENIED = "ingest_denied"
    INGEST_FAILED = "ingest_failed"
    FUZZY_MATCHED = "fuzzy_matched"
    LISTED = "listed"
    ASKED = "asked"
    SEARCHED = "searched"
    ANSWERED = "answered"
    ANSWER_FAILED = "answer_failed"
    EMBED_FAILED = "embed_failed"
    DELETED = "deleted"
    DELETE_DENIED = "delete_denied"
    ERROR = "error"


_MAX_FIELD_CHARS = 2000


class AuditLog:
    def __init__(self, path: Path | None) -> None:
        self._path = path
        self._lock = threading.Lock()

    def preflight(self) -> bool:
        if self._path is None:
            return True
        self._warn_if_world_readable()
        if self._writable():
            return True
        print(
            f"[audit] WARNING: {self._path} is not writable; falling back to stderr. "
            "Check that the audit mount is owned by the uid this server runs as.",
            file=sys.stderr,
            flush=True,
        )
        return False

    def write(self, event: Event, **fields: Any) -> None:
        record = {
            "ts": datetime.now(UTC).isoformat(),
            "event": str(event),
            **{k: _clip(v) for k, v in fields.items()},
        }
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            if self._path is not None and self._append(line):
                return
            print(f"[audit] {line}", file=sys.stderr, flush=True)

    def _warn_if_world_readable(self) -> None:
        parent = self._path.parent  # type: ignore[union-attr]
        try:
            mode = parent.stat().st_mode
        except OSError:
            return
        if mode & 0o077:
            print(
                f"[audit] WARNING: {parent} is mode {stat.S_IMODE(mode):04o}; "
                "use mode 700.",
                file=sys.stderr,
                flush=True,
            )

    def _writable(self) -> bool:
        try:
            with self._path.open("a", encoding="utf-8"):  # type: ignore[union-attr]
                return True
        except OSError:
            return False

    def _append(self, line: str) -> bool:
        try:
            with self._path.open("a", encoding="utf-8") as fh:  # type: ignore[union-attr]
                fh.write(line + "\n")
        except OSError:
            return False
        return True


def _clip(value: Any) -> Any:
    if isinstance(value, str) and len(value) > _MAX_FIELD_CHARS:
        return value[:_MAX_FIELD_CHARS] + "...[clipped]"
    return value
