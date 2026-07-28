"""Shared server bootstrap: env parsing, audit wiring, transport startup.

Adapted from sandbox-mcp's servers/_runtime.py -- same bearer-auth model
(mandatory once bound off loopback), same reasoning: this is the only thing
standing between the network and this server's tools.
"""

from __future__ import annotations

import os
import secrets
import sys
from collections.abc import Awaitable, Callable, MutableMapping
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from library_mcp.audit import AuditLog

_Scope = MutableMapping[str, Any]
_Receive = Callable[[], Awaitable[MutableMapping[str, Any]]]
_Send = Callable[[MutableMapping[str, Any]], Awaitable[None]]
_App = Callable[[_Scope, _Receive, _Send], Awaitable[None]]

_UNAUTHORIZED = b'{"error":"unauthorized"}'


def audit_from_env(var: str = "AUDIT_PATH") -> AuditLog:
    value = os.environ.get(var)
    log = AuditLog(Path(value) if value else None)
    log.preflight()
    return log


def int_from_env(var: str, default: int) -> int:
    raw = os.environ.get(var)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def str_from_env(var: str, default: str) -> str:
    return os.environ.get(var, default)


def build_app(name: str) -> FastMCP:
    return FastMCP(
        name,
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int_from_env("PORT", 8800),
        stateless_http=True,
    )


def require_bearer(app: _App, token: str) -> _App:
    expected = f"Bearer {token}".encode()

    async def wrapped(scope: _Scope, receive: _Receive, send: _Send) -> None:
        if scope.get("type") != "http":
            await app(scope, receive, send)
            return
        provided = b""
        for key, value in scope.get("headers") or []:
            if key.lower() == b"authorization":
                provided = value
                break
        if not secrets.compare_digest(provided, expected):
            await send(
                {
                    "type": "http.response.start",
                    "status": 401,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(_UNAUTHORIZED)).encode()),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": _UNAUTHORIZED})
            return
        await app(scope, receive, send)

    return wrapped


def serve(app: FastMCP) -> None:
    token = os.environ.get("AUTH_TOKEN", "").strip()
    if not token:
        print(
            "[transport] WARNING: AUTH_TOKEN is not set; this server accepts any caller "
            "that can reach its port. Safe only while it is bound to loopback.",
            file=sys.stderr,
            flush=True,
        )
        app.run(transport="streamable-http")
        return

    import uvicorn  # noqa: PLC0415

    uvicorn.run(
        require_bearer(app.streamable_http_app(), token),
        host=app.settings.host,
        port=app.settings.port,
        log_level=app.settings.log_level.lower(),
    )
