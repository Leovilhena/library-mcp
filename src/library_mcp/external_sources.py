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

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

import httpx

# Non-negotiable per Wikimedia's own API etiquette policy (§3.2) -- a
# missing/generic User-Agent is treated as a build-blocking bug, not a
# nice-to-have.
USER_AGENT = (
    "Pythia-KnowledgeGapResearch/1.0 "
    "(personal-use, single-user; contact: leosvilhena@icloud.com)"
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
        from urllib.parse import quote

        url = f"{self._rest_base}/page/summary/{quote(title, safe='')}"
        try:
            response = await client.get(url)
        except httpx.HTTPError as exc:
            msg = f"{self.name} summary fetch failed: {exc}"
            raise ExternalSourceError(msg) from exc
        if response.status_code == 404:
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


def default_sources() -> list[ExternalSource]:
    """Ordered: Wikipedia first (more likely to have a general answer),
    Wikiquote second (§3.1)."""
    return [WikipediaSource(), WikiquoteSource()]
