"""Tests for the ExternalSource interface + Wikipedia/Wikiquote
implementations (docs/planning/knowledge-gap-research.md §3, §9 build
step 2).

Two kinds of test, per the design doc's instruction: real API calls
against the live, free, no-key Wikipedia/Wikiquote endpoints (safe -- no
auth, no cost, and this suite runs at the same negligible volume the
design doc already reasons is fine for Wikimedia's own etiquette rules),
plus fixture-based/mocked tests that don't need network access, for CI.
"""

from __future__ import annotations

import httpx
import pytest

from library_mcp.external_sources import (
    USER_AGENT,
    ExternalSourceError,
    WikipediaSource,
    WikiquoteSource,
    default_sources,
    new_client,
)


# ---------------------------------------------------------------------------
# Real network tests -- free, no-key, safe to hit directly (§9 step 2)
# ---------------------------------------------------------------------------


@pytest.mark.network
async def test_wikipedia_health_check_against_the_real_api() -> None:
    source = WikipediaSource()
    async with new_client() as client:
        assert await source.health_check(client) is True


@pytest.mark.network
async def test_wikipedia_lookup_finds_a_real_page() -> None:
    source = WikipediaSource()
    async with new_client() as client:
        result = await source.lookup(client, "Marcus Aurelius")
    assert result is not None
    assert result.source == "wikipedia"
    assert "Aurelius" in result.title
    assert result.extract
    assert result.url.startswith("https://en.wikipedia.org/")


@pytest.mark.network
async def test_wikipedia_lookup_returns_none_for_pure_gibberish() -> None:
    source = WikipediaSource()
    async with new_client() as client:
        result = await source.lookup(
            client, "zzxqqvbnm nonexistent gibberish query asdkjfhalskdjfh"
        )
    # Not a hard guarantee (search engines are fuzzy), but this string is
    # deliberately unpronounceable garbage -- if this starts failing, the
    # search API's behavior changed in a way worth knowing about.
    assert result is None or "zzxqqvbnm" not in result.extract.lower()


@pytest.mark.network
async def test_wikiquote_health_check_against_the_real_api() -> None:
    source = WikiquoteSource()
    async with new_client() as client:
        assert await source.health_check(client) is True


@pytest.mark.network
async def test_wikiquote_lookup_finds_a_real_page() -> None:
    source = WikiquoteSource()
    async with new_client() as client:
        result = await source.lookup(client, "Marcus Aurelius quotes")
    assert result is not None
    assert result.source == "wikiquote"
    assert result.extract


def test_default_sources_are_ordered_wikipedia_first() -> None:
    sources = default_sources()
    assert [s.name for s in sources] == ["wikipedia", "wikiquote"]


def test_user_agent_names_the_app_and_contact() -> None:
    # §3.2: a missing/generic User-Agent is a build-blocking bug, not a
    # nice-to-have -- assert the real requirement, not just "is non-empty".
    assert "Pythia" in USER_AGENT
    assert "leosvilhena@icloud.com" in USER_AGENT


async def test_new_client_sends_the_user_agent_header() -> None:
    async with new_client() as client:
        assert client.headers["User-Agent"] == USER_AGENT


# ---------------------------------------------------------------------------
# Fixture-based / mocked tests -- offline, for CI (§9 step 2's instruction)
# ---------------------------------------------------------------------------


class _MockTransport(httpx.AsyncBaseTransport):
    def __init__(self, handler) -> None:
        self._handler = handler

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return self._handler(request)


def _client_with(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=_MockTransport(handler), headers={"User-Agent": USER_AGENT})


async def test_lookup_returns_none_when_search_has_no_hits() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"query": {"search": []}})

    source = WikipediaSource()
    async with _client_with(handler) as client:
        result = await source.lookup(client, "no such topic")
    assert result is None


async def test_lookup_returns_a_result_on_a_clean_search_and_summary(monkeypatch) -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if "action=query" in str(request.url):
            return httpx.Response(
                200, json={"query": {"search": [{"title": "Marcus Aurelius"}]}}
            )
        return httpx.Response(
            200,
            json={
                "extract": "Marcus Aurelius was a Roman emperor and Stoic philosopher.",
                "content_urls": {
                    "desktop": {"page": "https://en.wikipedia.org/wiki/Marcus_Aurelius"}
                },
            },
        )

    source = WikipediaSource()
    async with _client_with(handler) as client:
        result = await source.lookup(client, "who was Marcus Aurelius")
    assert result is not None
    assert result.title == "Marcus Aurelius"
    assert "Stoic philosopher" in result.extract
    assert result.url == "https://en.wikipedia.org/wiki/Marcus_Aurelius"


async def test_lookup_returns_none_when_the_matched_page_has_no_extract() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "action=query" in str(request.url):
            return httpx.Response(200, json={"query": {"search": [{"title": "Some Page"}]}})
        return httpx.Response(200, json={"extract": ""})

    source = WikipediaSource()
    async with _client_with(handler) as client:
        result = await source.lookup(client, "something")
    assert result is None


async def test_lookup_treats_a_404_summary_as_content_miss_not_infra_down() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "action=query" in str(request.url):
            return httpx.Response(200, json={"query": {"search": [{"title": "Vanished Page"}]}})
        return httpx.Response(404)

    source = WikipediaSource()
    async with _client_with(handler) as client:
        result = await source.lookup(client, "something")
    assert result is None  # content_miss, not a raised ExternalSourceError


async def test_search_http_5xx_raises_external_source_error() -> None:
    # §3.3: a transport-level failure (HTTP 5xx) must be classified
    # infra_down by the caller, never silently returned as "no result".
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    source = WikipediaSource()
    async with _client_with(handler) as client:
        with pytest.raises(ExternalSourceError):
            await source.lookup(client, "something")


async def test_summary_http_5xx_raises_external_source_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "action=query" in str(request.url):
            return httpx.Response(200, json={"query": {"search": [{"title": "A Page"}]}})
        return httpx.Response(500)

    source = WikipediaSource()
    async with _client_with(handler) as client:
        with pytest.raises(ExternalSourceError):
            await source.lookup(client, "something")


async def test_health_check_returns_false_on_transport_failure_not_a_raise() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    source = WikipediaSource()
    async with _client_with(handler) as client:
        assert await source.health_check(client) is False


async def test_health_check_returns_true_on_a_clean_siteinfo_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"query": {"general": {"sitename": "Wikipedia"}}})

    source = WikipediaSource()
    async with _client_with(handler) as client:
        assert await source.health_check(client) is True
