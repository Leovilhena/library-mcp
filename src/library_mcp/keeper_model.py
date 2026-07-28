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


class ReasoningError(Exception):
    """The reasoning call failed."""


@dataclass(frozen=True, slots=True)
class AnswerDecision:
    text: str


@dataclass(frozen=True, slots=True)
class SearchDecision:
    query: str


Decision = AnswerDecision | SearchDecision


class ReasoningClient:
    def __init__(self, base_url: str, model: str, timeout_seconds: float) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout_seconds

    async def decide(self, question: str, context: str, remaining_searches: int) -> Decision:
        prompt = _PROMPT_TEMPLATE.format(
            question=question, context=context, remaining=remaining_searches
        )
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
        text = str(data.get("response", "")).strip()
        return _parse_decision(text, remaining_searches)


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
