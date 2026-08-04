"""External research providers: a narrow, swappable interface, one
implementation per source (docs/planning/knowledge-gap-research.md §3).

Held to the same discipline as `EmbeddingClient`/`ReasoningClient` -- a
protocol, not a Wikipedia script, so Wikidata/Gutenberg/SEP/DuckDuckGo
(all named candidates, none built for v1, §9's Non-goals) are additive
later: implement `lookup()`+`health_check()`, register the instance, done.

TRANSPORT VS CONTENT (§3.3)
----------------------------
`lookup()` and `health_check()` raise `ExternalSourceError` ONLY for a
real transport/parsing failure (connection refused, DNS failure, timeout,
HTTP 5xx) -- the caller (the nightly orchestration module) classifies that
as `infra_down`. A clean response that simply doesn't address the question
returns `None` from `lookup()`, never an exception -- the caller classifies
that as `content_miss`. This mirrors the classification `sandbox_mcp`'s
`fetcher.py` already makes at the transport layer (`httpx.HTTPError` vs. a
successful response whose status is inspected separately), applied here as
an existing pattern, not a new invention.

ETIQUETTE (§3.2)
-----------------
Wikimedia's API policy requires a descriptive User-Agent naming the
application and contact information -- anonymous/generic User-Agents get
rate-limited or blocked. Set once, shared across every source
implementation in this module, never per-call.
"""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from http import HTTPStatus
from typing import Protocol, runtime_checkable
from urllib.parse import quote

import httpx

# Non-negotiable per Wikimedia's own API etiquette policy (§3.2) -- a
# missing/generic User-Agent is treated as a build-blocking bug, not a
# nice-to-have.
USER_AGENT = (
    "Pythia-KnowledgeGapResearch/1.0 (personal-use, single-user; contact: leosvilhena@icloud.com)"
)

_EXTRACT_MAX_CHARS = 4000
_DEFAULT_TIMEOUT = 10.0
_HEALTH_TIMEOUT = 5.0


class ExternalSourceError(Exception):
    """A real transport-level failure -- connection refused, DNS failure,
    timeout, or an HTTP 5xx response. The caller treats this as "the
    source is unavailable right now", never as a knowledge-gap verdict."""


@dataclass(frozen=True, slots=True)
class ExternalResult:
    source: str  # "wikipedia" | "wikiquote" | ... -- stored verbatim as provenance
    title: str  # page/article title actually matched
    url: str  # canonical URL, for Leo to check the source himself
    extract: str  # the fetched text, bounded
    fetched_at: str


@runtime_checkable
class ExternalSource(Protocol):
    name: str

    async def health_check(self, client: httpx.AsyncClient) -> bool:
        """One cheap, known-good request (§3.3's once-per-run probe).
        Raises `ExternalSourceError` only never -- returns False on any
        failure so the caller can log a clean `ok` bool per run without a
        try/except at every call site."""
        ...

    async def lookup(self, client: httpx.AsyncClient, question: str) -> ExternalResult | None:
        """Return the best matching passage for this question, or None if
        nothing plausible was found (`content_miss`). Raises
        `ExternalSourceError` for a real transport/parsing failure
        (`infra_down`) -- never for "no result"."""
        ...


def new_client(timeout: float = _DEFAULT_TIMEOUT) -> httpx.AsyncClient:
    """Shared client factory: one place the User-Agent is set, so every
    call site gets it for free rather than repeating it per-request."""
    return httpx.AsyncClient(timeout=timeout, headers={"User-Agent": USER_AGENT})


class _MediaWikiSource:
    """Shared implementation for Wikipedia/Wikiquote -- same MediaWiki
    software family, same two-step shape: a search API call to map a
    natural-language question onto a page title (the common case -- Leo's
    question is not a page title), then the REST summary endpoint for a
    clean extract."""

    def __init__(self, name: str, api_base: str, rest_base: str) -> None:
        self.name = name
        self._api_base = api_base.rstrip("/")
        self._rest_base = rest_base.rstrip("/")

    async def health_check(self, client: httpx.AsyncClient) -> bool:
        try:
            response = await client.get(
                self._api_base,
                params={"action": "query", "meta": "siteinfo", "format": "json"},
                timeout=_HEALTH_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError):
            return False
        return isinstance(data, dict) and "query" in data

    async def _search_title(self, client: httpx.AsyncClient, question: str) -> str | None:
        try:
            response = await client.get(
                self._api_base,
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": question,
                    "srlimit": 1,
                    "format": "json",
                },
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as exc:
            msg = f"{self.name} search failed: {exc}"
            raise ExternalSourceError(msg) from exc
        except ValueError as exc:
            msg = f"{self.name} search returned unparseable JSON: {exc}"
            raise ExternalSourceError(msg) from exc

        hits = ((data.get("query") or {}).get("search")) or []
        if not hits:
            return None
        title = hits[0].get("title")
        return str(title) if title else None

    async def _summary(self, client: httpx.AsyncClient, title: str) -> tuple[str, str] | None:
        """Returns (extract, canonical_url), or None if the page has no
        usable summary (a search hit with no real content -- content_miss,
        not a transport failure)."""
        url = f"{self._rest_base}/page/summary/{quote(title, safe='')}"
        try:
            response = await client.get(url)
        except httpx.HTTPError as exc:
            msg = f"{self.name} summary fetch failed: {exc}"
            raise ExternalSourceError(msg) from exc
        if response.status_code == HTTPStatus.NOT_FOUND:
            # A search hit whose page vanished/was renamed -- a real,
            # if unusual, content miss, not an outage.
            return None
        try:
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as exc:
            msg = f"{self.name} summary fetch failed: {exc}"
            raise ExternalSourceError(msg) from exc
        except ValueError as exc:
            msg = f"{self.name} summary returned unparseable JSON: {exc}"
            raise ExternalSourceError(msg) from exc

        extract = str(data.get("extract") or "").strip()
        if not extract:
            return None
        canonical = (
            (data.get("content_urls") or {}).get("desktop", {}).get("page")
            or data.get("canonicalurl")
            or url
        )
        return extract[:_EXTRACT_MAX_CHARS], str(canonical)

    async def lookup(self, client: httpx.AsyncClient, question: str) -> ExternalResult | None:
        title = await self._search_title(client, question)
        if title is None:
            return None
        summary = await self._summary(client, title)
        if summary is None:
            return None
        extract, canonical_url = summary
        return ExternalResult(
            source=self.name,
            title=title,
            url=canonical_url,
            extract=extract,
            fetched_at=datetime.now(UTC).isoformat(),
        )


class WikipediaSource(_MediaWikiSource):
    def __init__(self) -> None:
        super().__init__(
            name="wikipedia",
            api_base="https://en.wikipedia.org/w/api.php",
            rest_base="https://en.wikipedia.org/api/rest_v1",
        )


class WikiquoteSource(_MediaWikiSource):
    """Most useful for attributed-quote questions ("what did X say about
    Y") -- a meaningfully large fraction of book-adjacent questions given
    the library's current (philosophy-heavy) contents, per §3.2."""

    def __init__(self) -> None:
        super().__init__(
            name="wikiquote",
            api_base="https://en.wikiquote.org/w/api.php",
            rest_base="https://en.wikiquote.org/api/rest_v1",
        )


# Hard wall-clock cap for the whole subprocess round-trip. Enforced by the
# parent (`asyncio.wait_for` + kill on expiry), not by DDGS's own
# per-request `timeout=` alone -- ddgs's internal multi-engine retry loop
# has no overall cap of its own, so a slow/rate-limited response could hang
# past any single request's timeout.
_DDGS_SUBPROCESS_TIMEOUT = 20.0
_DDGS_MAX_RESULTS = 3


class DdgsSource:
    """General web search via DuckDuckGo (the `ddgs` package) -- the
    named-but-unbuilt-for-v1 candidate this module's own docstring
    anticipated. Unlike Wikipedia/Wikiquote, has no fixed-domain REST API
    to call directly, so `lookup()` shells out to `_ddgs_worker.py` in an
    isolated subprocess with a hard, killable timeout rather than calling
    `ddgs` in-process (see that module's docstring for why: its native
    HTTP client can block while holding the GIL, which a thread-pool
    timeout cannot reliably cancel).

    The `client: httpx.AsyncClient` parameter on both methods is unused --
    kept only so this still satisfies `ExternalSource`'s shared shape,
    since the orchestration layer (gap_research.py) opens one client and
    passes it to every registered source uniformly.
    """

    name = "web"

    async def health_check(self, client: httpx.AsyncClient) -> bool:  # noqa: ARG002
        try:
            await self._search("test", max_results=1)
        except ExternalSourceError:
            return False
        return True

    async def lookup(self, client: httpx.AsyncClient, question: str) -> ExternalResult | None:  # noqa: ARG002
        hits = await self._search(question, max_results=_DDGS_MAX_RESULTS)
        if not hits:
            return None
        extract = "\n\n".join(
            f"{hit['title']}: {hit['body']}".strip(": ")
            for hit in hits
            if hit.get("title") or hit.get("body")
        )[:_EXTRACT_MAX_CHARS]
        if not extract:
            return None
        return ExternalResult(
            source=self.name,
            title=question,
            url=hits[0].get("url", ""),
            extract=extract,
            fetched_at=datetime.now(UTC).isoformat(),
        )

    async def _search(self, query: str, max_results: int) -> list[dict[str, str]]:
        """Raises `ExternalSourceError` for any transport/subprocess
        failure -- the worker starting, timing out, or exiting non-zero.
        An empty (but successful) result list is a real content_miss,
        returned as `[]`, never an exception."""
        request = json.dumps({"query": query, "max_results": max_results})
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "library_mcp._ddgs_worker",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
        except OSError as exc:
            msg = f"web search worker failed to start: {exc}"
            raise ExternalSourceError(msg) from exc

        try:
            stdout, _ = await asyncio.wait_for(
                proc.communicate(request.encode()), timeout=_DDGS_SUBPROCESS_TIMEOUT
            )
        except TimeoutError as exc:
            proc.kill()
            await proc.wait()
            msg = f"web search timed out after {_DDGS_SUBPROCESS_TIMEOUT}s"
            raise ExternalSourceError(msg) from exc

        try:
            envelope = json.loads(stdout.decode())
        except (ValueError, UnicodeDecodeError) as exc:
            msg = f"web search worker returned invalid output: {exc}"
            raise ExternalSourceError(msg) from exc

        if not isinstance(envelope, dict) or not envelope.get("ok"):
            error = envelope.get("error") if isinstance(envelope, dict) else envelope
            msg = f"web search failed: {error}"
            raise ExternalSourceError(msg)
        results = envelope.get("results")
        return results if isinstance(results, list) else []


def default_sources() -> list[ExternalSource]:
    """Ordered: Wikipedia first (more likely to have a general answer),
    Wikiquote second (§3.1), general web search third -- the broadest,
    least-curated source, tried only after the two more targeted ones."""
    return [WikipediaSource(), WikiquoteSource(), DdgsSource()]
