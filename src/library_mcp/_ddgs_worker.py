"""DDGS (DuckDuckGo) search subprocess worker.

Runs the actual `ddgs` call in an isolated child process so a hang in its
underlying native HTTP client (`primp`) can be killed at the OS level by
the parent, rather than freezing the process. `primp` can block inside
native code while holding the GIL -- a `ThreadPoolExecutor` + timeout
cannot reliably cancel that, since the waiting thread never reacquires
the GIL to notice the timeout. See hermes-agent's
`plugins/web/ddgs/provider.py` (issue #68096) for the incident this
defends against; this worker is deliberately a smaller version of the
same fix, since library-mcp has no interrupt/test-hook machinery to
carry over.

Invoked as `python -m library_mcp._ddgs_worker`. Reads one JSON request
from stdin, writes one JSON envelope to stdout, then exits.

Request:  {"query": str, "max_results": int}
Envelope: {"ok": true, "results": [{"title": str, "url": str, "body": str}, ...]}
          {"ok": false, "error": str}
"""

from __future__ import annotations

import json
import sys

from ddgs import DDGS


def main() -> int:
    try:
        request = json.loads(sys.stdin.read())
        query = str(request["query"])
        max_results = max(1, int(request.get("max_results", 3)))
    except Exception as exc:
        json.dump({"ok": False, "error": f"invalid request: {exc}"}, sys.stdout)
        return 2

    try:
        with DDGS(timeout=10) as client:
            hits = list(client.text(query, max_results=max_results))
        results = [
            {
                "title": str(hit.get("title", "")),
                "url": str(hit.get("href") or hit.get("url") or ""),
                "body": str(hit.get("body", "")),
            }
            for hit in hits[:max_results]
        ]
        json.dump({"ok": True, "results": results}, sys.stdout)
        return 0
    except Exception as exc:
        json.dump({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, sys.stdout)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
