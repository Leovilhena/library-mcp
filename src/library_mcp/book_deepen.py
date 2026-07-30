"""Step one of knowledge-gap research: try harder against the books we
already have before ever touching a network call outside the local stack
(docs/planning/knowledge-gap-research.md §2, §9 build step 1).

A second `_ask()`-shaped pass over the same `knowledge.db`, deliberately
different from the live `ask_library` path in exactly the two dimensions
most likely to have caused the original miss: a wider `top_k` and a better
reasoning model than the live path can afford to keep resident alongside
the frontier model (`gemma3:1b`). This reuses `library_mcp.store`'s
`open_store()`/`search()` and the same `EmbeddingClient`/`ReasoningClient`
`keeper_server.py` already uses -- not a fresh reimplementation of the
search-and-reason loop, a second configuration of it.

Runs off the live request path (the nightly job, §9 step 6), so a wider
top_k and a bigger model are pure cost with no latency budget to protect --
the opposite trade `KeeperPolicy`'s defaults make for the interactive tool.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from library_mcp.audit import AuditLog, Event
from library_mcp.embedding import EmbeddingClient, EmbeddingError
from library_mcp.keeper_model import AnswerDecision, ReasoningClient, ReasoningError, SearchDecision
from library_mcp.store import SearchResult, search

# Candidates named in the design doc (§2). 8b first: 14b is already the
# nightly job's primary reasoning model for reflex-memory's own pass
# sharing this same VRAM window (reflex/reasoning.py), and running both
# passes at 14b back-to-back is a bigger time cost than this step's own
# marginal value justifies without real data showing 8b isn't enough.
DEFAULT_MODEL = "deepseek-r1:8b"
DEFAULT_TOP_K = 20
# Same bound `KeeperPolicy.max_searches` enforces on the live path --
# widening top_k/upgrading the model doesn't mean removing the loop's
# ceiling (§2).
DEFAULT_MAX_SEARCHES = 3
DEFAULT_MAX_CONTEXT_CHARS = 6000


@dataclass(frozen=True, slots=True)
class DeepenConfig:
    ollama_base_url: str
    embedding_model: str = "nomic-embed-text"
    reasoning_model: str = DEFAULT_MODEL
    embed_timeout_seconds: float = 30.0
    reasoning_timeout_seconds: float = 180.0
    top_k: int = DEFAULT_TOP_K
    max_searches: int = DEFAULT_MAX_SEARCHES
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS


def _format_context(results: list[SearchResult], max_chars: int) -> str:
    parts = []
    used = 0
    for r in results:
        entry = f"[{r.book_title} -- {r.section or 'unknown section'}]\n{r.text}\n"
        if used + len(entry) > max_chars:
            break
        parts.append(entry)
        used += len(entry)
    return "\n".join(parts) if parts else "(no matching passages found)"


async def deepen(
    conn,
    audit: AuditLog,
    gap_id: int,
    question: str,
    config: DeepenConfig,
) -> str | None:
    """Run the widened search+reason pass for one gap.

    Returns the answer text if the deepen pass settled on a confident
    answer with real backing results, or None if it came back empty or the
    reasoner still couldn't settle -- mirroring `_ask()`'s own
    "AnswerDecision with zero backing results = no real answer" signal
    (§0), but this function never writes to `knowledge_gaps` itself; the
    caller (nightly orchestration, §9 step 6) owns resolution bookkeeping.
    """
    audit.write(
        Event.BOOK_DEEPEN_ATTEMPTED, gap_id=gap_id, top_k=config.top_k, model=config.reasoning_model
    )

    embedder = EmbeddingClient(
        config.ollama_base_url, config.embedding_model, config.embed_timeout_seconds
    )
    reasoner = ReasoningClient(
        config.ollama_base_url, config.reasoning_model, config.reasoning_timeout_seconds
    )

    query = question
    all_results: list[SearchResult] = []
    seen_queries: set[str] = set()
    answer: str | None = None

    for attempt in range(config.max_searches):
        if query in seen_queries:
            break
        seen_queries.add(query)
        try:
            query_vector = await embedder.embed(query)
        except EmbeddingError:
            break

        results = search(conn, query_vector, config.top_k)
        new_text = {r.text for r in results}
        all_results = results + [r for r in all_results if r.text not in new_text]

        context = _format_context(all_results, config.max_context_chars)
        remaining = config.max_searches - attempt - 1
        try:
            decision = await reasoner.decide(query, context, remaining)
        except ReasoningError:
            break

        if isinstance(decision, AnswerDecision):
            if all_results:
                answer = decision.text
            break
        if isinstance(decision, SearchDecision):
            query = decision.query
            continue

    if answer is not None:
        audit.write(Event.BOOK_DEEPEN_RESOLVED, gap_id=gap_id)
        return answer

    audit.write(Event.BOOK_DEEPEN_NO_ANSWER, gap_id=gap_id)
    return None


def now_iso() -> str:
    return datetime.now(UTC).isoformat()
