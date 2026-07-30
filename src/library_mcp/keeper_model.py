"""The keeper's small reasoning model: decide 'answer' or 'search again'.

This is the bounded internal loop from the plan -- the frontier model calls
`ask_library` once and gets one synthesized answer back; the back-and-forth
for connected/multi-hop knowledge happens in here, not in the frontier's own
tool-calling, because that model's tool-calling is only 40-60% reliable and
should not be trusted to orchestrate a multi-turn search itself.

Same non-configurable-URL reasoning as embedding.py: the base URL comes only
from policy.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import httpx

_PROMPT_TEMPLATE = """You are a research assistant answering strictly from the passages below. \
Do not use outside knowledge -- if the passages don't answer the question, say so.

Question: {question}

Retrieved passages:
{context}

Respond with exactly one of these two forms, nothing else:
ANSWER: <your answer, citing book and section for each claim>
SEARCH: <a different, more specific search query that might find what's missing>

You have {remaining} search attempt(s) left after this one. If this is your \
last attempt, you must respond with ANSWER using whatever you have, even if \
it means saying the passages don't cover it.
"""

# Book structuring (docs/planning/book-structuring.md §3, §9 step 2). Bounded
# input (one chapter's text, not the whole book) -- the extractive task §3
# argues is tractable for a mid-size local model, unlike whole-book
# synthesis. Deliberately asks for "based only on this chapter": a glossary
# entry is Pythia's own synthesis (store.py's GlossaryEntry docstring), but
# it should still be traceable back to the source chapter, not the model's
# outside training knowledge about the term.
_GLOSSARY_PROMPT_TEMPLATE = """You extract a short glossary of key terms from one chapter of a \
book. Read the chapter text below and identify the terms it defines, explains, or centers on \
-- not just any word that appears, but concepts the chapter is actually teaching or relying on \
-- along with a concise definition based only on what this chapter itself says.

Chapter text:
{chapter_text}

Respond with one block per term, in exactly this format, and nothing else:
TERM: <term>
DEFINITION: <one to three sentence definition, based only on this chapter's text>

List at most {max_terms} terms, the most important ones first. If this chapter does not \
clearly define or center on any terms, respond with exactly:
NONE
"""

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.S)
_TERM_RE = re.compile(r"^\s*TERM:\s*(.+?)\s*$", re.I)
_DEFINITION_RE = re.compile(r"^\s*DEFINITION:\s*(.+?)\s*$", re.I)
# Real deepseek-r1:8b behavior, found live during this build's own manual
# verification against a real chapter (~83k chars -- a real spine item's
# full reconstructed text, not a short fixture): on longer input the model
# kept the DEFINITION: line but dropped the TERM: prefix in favor of a
# markdown-bold, optionally-numbered header line instead (e.g.
# "**1. imagination**"). Same drift-toward-its-own-preferred-shape failure
# class as the JSON-array fallback above, just a third shape. Matched only
# when the line-based TERM:/DEFINITION: parse doesn't already claim it, via
# the same fallback ordering as the JSON case.
_BOLD_TERM_RE = re.compile(r"^\s*\*\*\s*(?:\d+[.):]\s*)?(.+?)\s*\*\*\s*$")
# Real deepseek-r1:8b output, found live during this build's own manual
# verification against the real library: despite the prompt's explicit
# TERM:/DEFINITION: instruction, one real response ignored the format
# entirely and returned a fenced JSON array of {"term": ..., "definition":
# ...} objects instead -- the same "small model drifts toward its own
# preferred shape" failure class _find_marker's docstring already
# documents for gemma3:1b, just a different drift target. Matches the
# LAST such bracketed block in the text (a model's closing JSON answer,
# not an example embedded earlier in its own reasoning prose).
_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.S)


class ReasoningError(Exception):
    """The reasoning call failed."""


@dataclass(frozen=True, slots=True)
class AnswerDecision:
    text: str


@dataclass(frozen=True, slots=True)
class SearchDecision:
    query: str


Decision = AnswerDecision | SearchDecision


@dataclass(frozen=True, slots=True)
class GlossaryCandidate:
    term: str
    definition: str


class ReasoningClient:
    def __init__(self, base_url: str, model: str, timeout_seconds: float) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout_seconds

    async def _generate(self, prompt: str) -> str:
        """The shared HTTP-to-Ollama plumbing both `decide()` and
        `extract_glossary_terms()` use -- one client, two prompt shapes
        (docs/planning/book-structuring.md §9 step 2: "reusing
        ReasoningClient's existing HTTP-to-Ollama plumbing... not a new
        client")."""
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"{self._base_url}/api/generate",
                    json={
                        "model": self._model,
                        "prompt": prompt,
                        "stream": False,
                        "keep_alive": "30s",
                    },
                )
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPError as exc:
            msg = f"reasoning request failed: {exc}"
            raise ReasoningError(msg) from exc
        return str(data.get("response", "")).strip()

    async def decide(self, question: str, context: str, remaining_searches: int) -> Decision:
        prompt = _PROMPT_TEMPLATE.format(
            question=question, context=context, remaining=remaining_searches
        )
        text = await self._generate(prompt)
        return _parse_decision(text, remaining_searches)

    async def extract_glossary_terms(
        self, chapter_text: str, max_terms: int = 12
    ) -> list[GlossaryCandidate]:
        """One call, one chapter's text in, `(term, definition)` pairs out
        (§9 step 2). Never raises on a malformed/empty model response --
        only on a real transport failure (`ReasoningError`, same as
        `decide()`) -- a chapter that produces zero clean pairs is the
        caller's signal to mark it `status='failed'` rather than silently
        empty (§9 step 2's own "make failure visible" instinct)."""
        prompt = _GLOSSARY_PROMPT_TEMPLATE.format(chapter_text=chapter_text, max_terms=max_terms)
        text = await self._generate(prompt)
        return _parse_glossary(text)


def _parse_glossary(text: str) -> list[GlossaryCandidate]:
    """Tolerant line-based parsing, same format-fragility instinct as
    `_find_marker` below: deepseek-r1 wraps its answer in a `<think>` block
    (reflex/reasoning.py's own `_THINK_BLOCK` precedent), so that's stripped
    first. A `TERM:` line (or a markdown-bold header line, `_BOLD_TERM_RE`
    -- a real observed alternate shape) starts a new candidate; the next
    `DEFINITION:` line completes it. A term with no following `DEFINITION:`
    (or vice versa) is silently dropped rather than guessed at -- a
    half-formed pair is worse than no pair, per this design's "confidently
    wrong" risk (§10). Falls back to `_parse_glossary_json_fallback` only
    if this line-based pass finds nothing at all.
    """
    cleaned = _THINK_BLOCK_RE.sub("", text).strip()
    if not cleaned or cleaned.strip().upper() == "NONE":
        return []
    candidates: list[GlossaryCandidate] = []
    pending_term: str | None = None
    for line in cleaned.splitlines():
        term_match = _TERM_RE.match(line)
        if term_match:
            pending_term = term_match.group(1).strip()
            continue
        bold_match = _BOLD_TERM_RE.match(line)
        if bold_match:
            pending_term = bold_match.group(1).strip()
            continue
        definition_match = _DEFINITION_RE.match(line)
        if definition_match and pending_term:
            definition = definition_match.group(1).strip()
            if pending_term and definition:
                candidates.append(GlossaryCandidate(term=pending_term, definition=definition))
            pending_term = None
    if candidates:
        return candidates
    return _parse_glossary_json_fallback(cleaned)


def _parse_glossary_json_fallback(cleaned: str) -> list[GlossaryCandidate]:
    """Only reached when the line-based TERM:/DEFINITION: parse above found
    nothing -- see the `_JSON_ARRAY_RE` comment for why this exists."""
    matches = _JSON_ARRAY_RE.findall(cleaned)
    if not matches:
        return []
    try:
        parsed = json.loads(matches[-1])
    except (json.JSONDecodeError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    candidates: list[GlossaryCandidate] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        term = str(item.get("term") or "").strip()
        definition = str(item.get("definition") or "").strip()
        if term and definition:
            candidates.append(GlossaryCandidate(term=term, definition=definition))
    return candidates


def _find_marker(text: str, marker: str) -> str | None:
    """Find `marker:` anywhere in the text, not just as a strict prefix.

    Empirically necessary: gemma3:1b's very first real test run returned
    prose *followed by* a trailing "SEARCH: async" rather than the
    requested single clean form. A strict `str.startswith` check missed it
    entirely and silently treated the blended text as a final answer. This
    scans for the marker anywhere and takes everything after it, which is
    the more common failure shape for a small model given a two-format
    instruction -- it drifts toward one, but doesn't drop it.
    """
    idx = text.upper().find(marker.upper())
    if idx == -1:
        return None
    return text[idx + len(marker) :].strip()


def _parse_decision(text: str, remaining_searches: int) -> Decision:
    if remaining_searches <= 0:
        # Last attempt already told the model to always answer; if it still
        # didn't, don't loop forever on an unreliable small model -- return
        # whatever text it gave rather than erroring or looping again.
        answer = _find_marker(text, "ANSWER:")
        return AnswerDecision(text=answer or text)

    # Check SEARCH first: when a response blends both, a further search is
    # the more conservative choice (it can still lead to an answer next
    # round; the reverse -- returning a premature answer -- can't recover).
    search_query = _find_marker(text, "SEARCH:")
    if search_query:
        return SearchDecision(query=search_query)
    answer = _find_marker(text, "ANSWER:")
    if answer:
        return AnswerDecision(text=answer)
    # Model didn't follow the format at all -- safe failure mode is to treat
    # the raw text as the answer rather than fail the whole request.
    return AnswerDecision(text=text)
