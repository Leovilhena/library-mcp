"""Glossary-extraction prompt/parsing tests
(docs/planning/book-structuring.md §3, §9 build step 2).

`_parse_glossary` is tested directly against fixture text (no Ollama
needed) -- the format-fragility case that matters most here, since
`deepseek-r1:8b` wraps output in a `<think>` block and this stack's small
models are only 40-60% reliable at strict formatting.
`test_extract_glossary_terms_against_real_ollama`
is a real integration test (skipped if Ollama is unreachable) confirming
the live model actually produces parseable pairs, not just that the parser
handles a fixture correctly.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from library_mcp.keeper_model import (
    GlossaryCandidate,
    ReasoningClient,
    ReasoningError,
    _parse_glossary,
)

# ---------------------------------------------------------------------------
# _parse_glossary: pure-function tests against fixture text
# ---------------------------------------------------------------------------


def test_parse_glossary_extracts_clean_pairs() -> None:
    text = (
        "TERM: Recursion\n"
        "DEFINITION: A function that calls itself to solve smaller instances "
        "of the same problem.\n"
        "TERM: Base case\n"
        "DEFINITION: The condition that stops a recursive function from "
        "calling itself forever.\n"
    )
    result = _parse_glossary(text)
    assert result == [
        GlossaryCandidate(
            term="Recursion",
            definition="A function that calls itself to solve smaller instances "
            "of the same problem.",
        ),
        GlossaryCandidate(
            term="Base case",
            definition="The condition that stops a recursive function from calling itself forever.",
        ),
    ]


def test_parse_glossary_strips_a_think_block_first() -> None:
    # deepseek-r1's own wrapping behavior (reflex/reasoning.py's
    # _THINK_BLOCK precedent) -- must not be mistaken for real content.
    text = (
        "<think>Let me consider which terms are important here...</think>\n"
        "TERM: Asyncio\n"
        "DEFINITION: Python's library for writing concurrent code using the "
        "async/await syntax.\n"
    )
    result = _parse_glossary(text)
    assert result == [
        GlossaryCandidate(
            term="Asyncio",
            definition="Python's library for writing concurrent code using the async/await syntax.",
        )
    ]


def test_parse_glossary_returns_empty_list_on_none_response() -> None:
    assert _parse_glossary("NONE") == []
    assert _parse_glossary("  none  \n") == []


def test_parse_glossary_returns_empty_list_on_empty_response() -> None:
    assert _parse_glossary("") == []
    assert _parse_glossary("   \n\n  ") == []


def test_parse_glossary_drops_a_term_with_no_following_definition() -> None:
    # A half-formed pair is worse than no pair (§10's "confidently wrong"
    # risk) -- silently dropped, not guessed at.
    text = "TERM: Orphaned term\nTERM: Recursion\nDEFINITION: A real one.\n"
    result = _parse_glossary(text)
    assert result == [GlossaryCandidate(term="Recursion", definition="A real one.")]


def test_parse_glossary_drops_a_definition_with_no_preceding_term() -> None:
    text = "DEFINITION: orphaned, no term before it\nTERM: Real\nDEFINITION: A real one.\n"
    result = _parse_glossary(text)
    assert result == [GlossaryCandidate(term="Real", definition="A real one.")]


def test_parse_glossary_accepts_a_numbered_bold_header_as_a_term_line() -> None:
    # Real deepseek-r1:8b behavior on a longer (~83k char) real chapter,
    # found live during this build's own manual verification: it kept the
    # DEFINITION: line but replaced "TERM: x" with a numbered bold header.
    text = (
        "**1. imagination**\n"
        "DEFINITION: Something indefinite with regard to which the soul is passive.\n"
        "**2. understanding**\n"
        "DEFINITION: The faculty that forms true ideas directly from its essence.\n"
    )
    result = _parse_glossary(text)
    assert result == [
        GlossaryCandidate(
            term="imagination",
            definition="Something indefinite with regard to which the soul is passive.",
        ),
        GlossaryCandidate(
            term="understanding",
            definition="The faculty that forms true ideas directly from its essence.",
        ),
    ]


def test_parse_glossary_accepts_a_plain_bold_header_without_a_number() -> None:
    text = "**Closure**\nDEFINITION: A function that captures variables from its enclosing scope.\n"
    result = _parse_glossary(text)
    assert result == [
        GlossaryCandidate(
            term="Closure",
            definition="A function that captures variables from its enclosing scope.",
        )
    ]


def test_parse_glossary_falls_back_to_a_json_array_response() -> None:
    # Real deepseek-r1:8b behavior, found live during this build's own
    # manual verification against the real library: despite the explicit
    # TERM:/DEFINITION: instruction, one real response ignored the format
    # and returned a fenced JSON array instead.
    text = (
        "Okay, here is a glossary:\n\n1. **Real Good**\n2. **True Happiness**\n\n"
        '```json\n[\n    {\n        "term": "real good",\n        '
        '"definition": "That which communicates itself through the intellect alone."\n    },\n'
        '    {\n        "term": "true happiness",\n        '
        '"definition": "Continuous, supreme joy derived from possessing the real '
        'good."\n    }\n]\n```'
    )
    result = _parse_glossary(text)
    assert result == [
        GlossaryCandidate(
            term="real good",
            definition="That which communicates itself through the intellect alone.",
        ),
        GlossaryCandidate(
            term="true happiness",
            definition="Continuous, supreme joy derived from possessing the real good.",
        ),
    ]


def test_parse_glossary_json_fallback_ignores_malformed_json() -> None:
    text = "Here's something that looks like JSON but isn't: [not valid json at all}"
    assert _parse_glossary(text) == []


def test_parse_glossary_json_fallback_skips_entries_missing_term_or_definition() -> None:
    text = (
        '[{"term": "only a term"}, {"definition": "only a definition"}, '
        '{"term": "complete", "definition": "has both"}]'
    )
    result = _parse_glossary(text)
    assert result == [GlossaryCandidate(term="complete", definition="has both")]


def test_parse_glossary_line_based_format_wins_over_json_fallback() -> None:
    # If the line-based TERM:/DEFINITION: parse already found real pairs,
    # the JSON fallback must never run (and must never double-count a
    # coincidental bracket elsewhere in the text).
    text = (
        "TERM: Recursion\nDEFINITION: A function that calls itself.\n"
        "Some stray text with [brackets] that is not JSON."
    )
    result = _parse_glossary(text)
    assert result == [
        GlossaryCandidate(term="Recursion", definition="A function that calls itself.")
    ]


def test_parse_glossary_is_tolerant_of_blended_prose() -> None:
    # gemma3:1b's own documented failure shape (_find_marker's docstring)
    # is prose blended with the expected format -- deepseek-r1 is more
    # reliable but the parser should still degrade gracefully on stray text
    # around the clean TERM:/DEFINITION: lines rather than erroring.
    text = (
        "Here is the glossary you asked for:\n"
        "TERM: Closure\n"
        "DEFINITION: A function that captures variables from its enclosing scope.\n"
        "I hope that helps!\n"
    )
    result = _parse_glossary(text)
    assert result == [
        GlossaryCandidate(
            term="Closure",
            definition="A function that captures variables from its enclosing scope.",
        )
    ]


# ---------------------------------------------------------------------------
# extract_glossary_terms: HTTP plumbing (mocked transport, no real Ollama)
# ---------------------------------------------------------------------------


def test_extract_glossary_terms_parses_a_mocked_response() -> None:
    async def _run() -> list[GlossaryCandidate]:
        client = ReasoningClient("http://fake:11434", "deepseek-r1:8b", 30.0)

        async def _fake_generate(prompt: str) -> str:
            assert "chapter text" in prompt.lower() or "Chapter text" in prompt
            return "TERM: Widget\nDEFINITION: A stand-in term used only in this test.\n"

        client._generate = _fake_generate  # type: ignore[method-assign]
        return await client.extract_glossary_terms("Some chapter text about widgets.")

    result = asyncio.run(_run())
    assert result == [
        GlossaryCandidate(term="Widget", definition="A stand-in term used only in this test.")
    ]


def test_extract_glossary_terms_raises_reasoning_error_on_transport_failure(monkeypatch) -> None:
    async def _run() -> None:
        client = ReasoningClient("http://fake:11434", "deepseek-r1:8b", 30.0)

        class _FailingAsyncClient:
            def __init__(self, *args, **kwargs) -> None:
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def post(self, *_args, **_kwargs):
                raise httpx.ConnectError("connection refused")

        monkeypatch.setattr("library_mcp.keeper_model.httpx.AsyncClient", _FailingAsyncClient)
        with pytest.raises(ReasoningError):
            await client.extract_glossary_terms("Some chapter text.")

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Real Ollama integration test -- skipped if unreachable, per the task's
# requirement that deepseek-r1:8b's real output is verified at least once,
# not only against a fixture.
# ---------------------------------------------------------------------------

_REAL_OLLAMA_URL = "http://172.27.224.1:11434"
_REAL_CHAPTER_TEXT = """
Chapter One: What Is Recursion?

Recursion is a technique in which a function solves a problem by calling
itself on smaller instances of the same problem. Every recursive function
needs a base case -- a condition under which it stops calling itself and
returns a direct answer, rather than recursing forever. Without a base
case, a recursive function will keep calling itself until the program runs
out of memory, a failure known as a stack overflow.

A classic example is computing a factorial: factorial(n) is n times
factorial(n - 1), with factorial(0) defined as 1 as the base case. Each
call to factorial(n) waits for factorial(n - 1) to return before it can
compute its own result, which is why a chain of recursive calls is
sometimes called a call stack.
"""


def _ollama_reachable() -> bool:
    try:
        response = httpx.get(f"{_REAL_OLLAMA_URL}/api/tags", timeout=3.0)
        return response.status_code == 200
    except httpx.HTTPError:
        return False


@pytest.mark.skipif(not _ollama_reachable(), reason="local Ollama not reachable")
def test_extract_glossary_terms_against_real_ollama() -> None:
    async def _attempt() -> list[GlossaryCandidate]:
        client = ReasoningClient(_REAL_OLLAMA_URL, "deepseek-r1:8b", 180.0)
        return await client.extract_glossary_terms(_REAL_CHAPTER_TEXT, max_terms=5)

    # Real-model output is not deterministic -- manual spot-checking during
    # this build showed deepseek-r1:8b usually returns clean TERM:/
    # DEFINITION: pairs for this fixture but occasionally returns a
    # response the parser can't extract anything from on a single call
    # (the same "a chapter that produces zero clean pairs" case
    # book_structure.py itself handles by marking that one chapter
    # status='failed' and moving on, per §9 step 2 -- not a bug). The
    # requirement this test exists to prove is that the model CAN produce
    # parseable pairs from real chapter text, not that every single call
    # succeeds -- so it retries a few times and requires at least one
    # attempt to produce real output before concluding the model/parser
    # pairing is broken.
    result: list[GlossaryCandidate] = []
    for _attempt_number in range(3):
        result = asyncio.run(_attempt())
        if result:
            break
    assert len(result) >= 1, "deepseek-r1:8b produced zero parseable pairs across 3 attempts"
    for candidate in result:
        assert candidate.term.strip()
        assert candidate.definition.strip()
