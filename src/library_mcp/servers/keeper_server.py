"""library-keeper MCP server: the specialized sub-agent.

Exposes exactly one tool, `ask_library`, to the frontier model. Everything
between "question in" and "answer out" -- embedding, searching, deciding
whether to search again for connected knowledge, synthesizing a cited answer
-- happens inside this process's own bounded loop, so the frontier's
tool-calling (only 40-60% reliable) never has to orchestrate a multi-turn
exchange itself.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime

from mcp.server.fastmcp import FastMCP

from library_mcp.audit import AuditLog, Event
from library_mcp.config import KeeperPolicy, PolicyError, load_keeper_policy, policy_path_from_env
from library_mcp.embedding import EmbeddingClient, EmbeddingError
from library_mcp.keeper_model import AnswerDecision, ReasoningClient, ReasoningError, SearchDecision
from library_mcp.runtime import audit_from_env, build_app, serve
from library_mcp.store import (
    GlossaryEntry,
    SearchResult,
    list_pending_followups as list_pending_followups_fn,
    lookup_glossary_term,
    mark_followups_delivered,
    open_store,
    record_knowledge_gap,
    search,
)
from library_mcp.store import list_knowledge_gaps as list_knowledge_gaps_fn

# Book structuring (docs/planning/book-structuring.md §4, §9 step 4): a
# cheap, deterministic pre-check for "does this question look like a
# term/definition lookup" -- written as a regex, not judged by the small
# reasoning model, per this stack's house rule (deterministic code over
# small-model judgment wherever a decision can be made without one). Only
# matches a short, single-concept phrasing ("what is X", "define X", "what
# does X mean") -- a longer, multi-clause question that happens to start
# with "what is" is exactly the case this must NOT fire on, since a wrong
# glossary hint injected into a broad question's context is more likely to
# mislead than help.
_TERM_LOOKUP_RE = re.compile(
    r"^\s*(?:what\s+(?:is|are)|define|what\s+does)\s+(?:an?\s+|the\s+)?(.+?)"
    r"(?:\s+mean)?\s*\??\s*$",
    re.I,
)
_TERM_LOOKUP_MAX_WORDS = 6


def _term_lookup_candidate(question: str) -> str | None:
    """Extract a candidate glossary term from `question`, or None if it
    doesn't look like a term/definition lookup. Deterministic, no model
    call -- see the module-level comment above `_TERM_LOOKUP_RE`."""
    match = _TERM_LOOKUP_RE.match(question)
    if not match:
        return None
    candidate = match.group(1).strip().strip("?").strip()
    if not candidate or len(candidate.split()) > _TERM_LOOKUP_MAX_WORDS:
        return None
    return candidate


@dataclass(frozen=True, slots=True)
class _Deps:
    policy: KeeperPolicy
    audit: AuditLog


def _format_context(
    results: list[SearchResult], max_chars: int, glossary_entry: GlossaryEntry | None = None
) -> str:
    """Additive, never a substitute (docs/planning/book-structuring.md §4,
    §9 step 4): a glossary hit is prepended as its own clearly labeled
    block, visually and structurally distinct from the `[{book_title} --
    {section}]` chunk blocks below it, so the reasoning model can never
    confuse Pythia's own synthesis for verbatim source text. Real chunk
    search results are always included in the same call, regardless of
    whether a glossary entry was found -- this function never chooses one
    or the other."""
    parts = []
    used = 0
    if glossary_entry is not None:
        entry = (
            "[Glossary entry -- Pythia's own synthesis, not verbatim book text -- "
            f"{glossary_entry.book_title}]\n{glossary_entry.term}: {glossary_entry.definition}\n"
        )
        parts.append(entry)
        used += len(entry)
    for r in results:
        entry = f"[{r.book_title} -- {r.section or 'unknown section'}]\n{r.text}\n"
        if used + len(entry) > max_chars:
            break
        parts.append(entry)
        used += len(entry)
    return "\n".join(parts) if parts else "(no matching passages found)"


async def _ask(deps: _Deps, question: str) -> str:
    policy = deps.policy
    conn = open_store(policy.db_path)

    # Book structuring's deterministic term-lookup pre-check (§4, §9 step
    # 4), run once, ahead of the first search round -- never gates or
    # replaces the real search below, it only adds one extra labeled
    # context block if it fires. `glossary_hit`/`glossary_term` on the
    # ASKED event give step 7's "does the pre-check actually fire on real
    # questions" observation something concrete to check in
    # keeper-audit/audit.jsonl without a new event type.
    term_candidate = _term_lookup_candidate(question)
    glossary_entry = lookup_glossary_term(conn, term_candidate) if term_candidate else None
    deps.audit.write(
        Event.ASKED,
        detail=question,
        glossary_hit=glossary_entry is not None,
        glossary_term=glossary_entry.term if glossary_entry else None,
    )

    embedder = EmbeddingClient(
        policy.ollama_base_url, policy.embedding_model, policy.embed_timeout_seconds
    )
    reasoner = ReasoningClient(
        policy.ollama_base_url, policy.reasoning_model, policy.reasoning_timeout_seconds
    )

    query = question
    all_results: list[SearchResult] = []
    seen_queries: set[str] = set()

    for attempt in range(policy.max_searches):
        if query in seen_queries:
            # The model asked for the same search again -- stop looping and
            # force a final answer rather than spin on a repeated query.
            break
        seen_queries.add(query)
        try:
            query_vector = await embedder.embed(query)
        except EmbeddingError as exc:
            deps.audit.write(Event.ANSWER_FAILED, detail=question, reason=f"embed failed: {exc}")
            return "I couldn't search the library right now (embedding call failed)."

        results = search(conn, query_vector, policy.top_k)
        deps.audit.write(Event.SEARCHED, detail=query, matches=len(results))
        # New results first so the most relevant recent search dominates the
        # context budget, then de-dupe by text.
        new_text = {r.text for r in results}
        all_results = results + [r for r in all_results if r.text not in new_text]

        context = _format_context(all_results, policy.max_context_chars, glossary_entry)
        remaining = policy.max_searches - attempt - 1
        try:
            decision = await reasoner.decide(query, context, remaining)
        except ReasoningError as exc:
            deps.audit.write(
                Event.ANSWER_FAILED, detail=question, reason=f"reasoning failed: {exc}"
            )
            return "I couldn't reason over the retrieved passages right now."

        if isinstance(decision, AnswerDecision):
            deps.audit.write(Event.ANSWERED, detail=question, searches=attempt + 1)
            if not all_results:
                # Real bug found by review: `_parse_decision` always returns
                # AnswerDecision once `remaining_searches` hits 0 (the last
                # attempt's prompt explicitly tells the model to answer
                # regardless), so this -- not the loop-exhaustion fallthrough
                # below -- is the actual reachable path for "we genuinely
                # have nothing" (the prompt tells the model to say so, which
                # produces a real AnswerDecision, not a SearchDecision). The
                # loop can never fall through on its own; only the
                # duplicate-query break below reaches that code at all.
                record_knowledge_gap(conn, question, "no_matches", datetime.now(UTC).isoformat())
            return decision.text
        if isinstance(decision, SearchDecision):
            query = decision.query
            continue

    # Only reachable via the duplicate-query break above -- `_parse_decision`
    # always returns AnswerDecision on the final iteration (remaining == 0),
    # so normal loop exhaustion never falls through to here. Label the
    # reason for what actually happened, not what the code used to assume.
    deps.audit.write(Event.ANSWER_FAILED, detail=question, reason="repeated query, no answer")
    record_knowledge_gap(conn, question, "repeated_query_no_answer", datetime.now(UTC).isoformat())
    return (
        "I searched the library but couldn't settle on a confident answer. "
        "Closest passages found:\n\n"
        f"{_format_context(all_results, policy.max_context_chars, glossary_entry)}"
    )


def build_server(policy: KeeperPolicy, audit: AuditLog) -> FastMCP:
    deps = _Deps(policy=policy, audit=audit)
    app = build_app("library-keeper")

    @app.tool(
        description=(
            "Ask a question grounded in the books that have been learned via the learn tool. "
            "Internally searches the shared knowledge base up to its own configured limit "
            "(possibly more than once, for connected topics across books) BEFORE returning, "
            "and returns one synthesized, cited answer -- broadening the search yourself by "
            "calling this tool again with a rephrased question is redundant and just doubles "
            "latency and failure exposure for no extra coverage. Call it once per question; "
            "only call it again in the same turn if the user asked about a genuinely different "
            "topic. Use this instead of answering from your own memory when the question is "
            "about specific book content."
        )
    )
    async def ask_library(question: str) -> str:
        return await _ask(deps, question)

    @app.tool(
        description=(
            "List real questions the knowledge base couldn't confidently answer -- things "
            "asked that turned up no matches, or where no confident answer could be settled "
            "on. Use this to see what's worth adding a book/source for, not for answering "
            "questions directly."
        )
    )
    def list_knowledge_gaps() -> str:
        conn = open_store(deps.policy.db_path)
        gaps = list_knowledge_gaps_fn(conn)
        if not gaps:
            return "No open knowledge gaps recorded."
        lines = [
            f'- "{g.question}" (asked {g.times_asked}x, last {g.last_asked_at}, reason: {g.reason})'
            for g in gaps
        ]
        return f"{len(gaps)} open knowledge gap(s):\n" + "\n".join(lines)

    # Knowledge-gap research (docs/planning/knowledge-gap-research.md §5.1,
    # §9 step 7): two internal delivery-bookkeeping tools for the
    # `knowledge_followup` gateway plugin's `pre_llm_call` hook. Both are
    # excluded from `mcp_servers.library_keeper.tools` in config.yaml so the
    # frontier model never sees them as callable -- the decision of what
    # gets surfaced and when has to stay fully deterministic (§0's house
    # rule), never left to the 40-60%-reliable frontier model's own
    # tool-calling judgment. The plugin reaches them via its own direct MCP
    # client session against this same server, not through the model's
    # tool-calling machinery (§5.1).

    @app.tool(
        description=(
            "Internal delivery-bookkeeping tool for the knowledge_followup "
            "gateway plugin's pre_llm_call hook (docs/planning/"
            "knowledge-gap-research.md §5) -- lists resolved-but-not-yet-"
            "mentioned followups (a deeper book re-search or an external "
            "lookup has since answered a question that previously failed). "
            "NOT intended for the frontier model to call on its own initiative "
            "-- excluded from mcp_servers.library_keeper.tools in config.yaml "
            "for exactly that reason (see below). Use list_knowledge_gaps "
            "instead for reviewing what remains genuinely open."
        )
    )
    def list_pending_followups(chat_id: str, limit: int = 2) -> str:
        conn = open_store(deps.policy.db_path)
        rows = list_pending_followups_fn(conn, chat_id, limit)
        if not rows:
            return "[]"
        return json.dumps(
            [
                {
                    "id": r.id,
                    "gap_id": r.gap_id,
                    "question": r.question,
                    "resolution_kind": r.resolution_kind,
                    "summary": r.summary,
                    "resolved_at": r.resolved_at,
                }
                for r in rows
            ],
            ensure_ascii=False,
        )

    @app.tool(
        description=(
            "Internal: mark one or more pending followups as delivered -- "
            "handed to the model to mention in a live turn, see "
            "docs/planning/knowledge-gap-research.md §5.2 for what 'delivered' "
            "does and doesn't verify. Idempotent: marking an already-delivered "
            "id again is a no-op. Same non-model-facing exclusion as "
            "list_pending_followups."
        )
    )
    def mark_followup_delivered(followup_ids: list[int]) -> str:
        conn = open_store(deps.policy.db_path)
        injected_at = datetime.now(UTC).isoformat()
        delivered = mark_followups_delivered(conn, followup_ids, injected_at)
        if delivered:
            rows = conn.execute(
                "SELECT id, gap_id, chat_id FROM pending_followups WHERE id IN "
                f"({','.join('?' * len(delivered))})",
                delivered,
            ).fetchall()
            for followup_id, gap_id, chat_id in rows:
                deps.audit.write(
                    Event.FOLLOWUP_INJECTED,
                    gap_id=int(gap_id),
                    followup_id=int(followup_id),
                    chat_id=chat_id,
                )
        return json.dumps({"delivered": delivered}, ensure_ascii=False)

    return app


def main() -> None:
    try:
        policy = load_keeper_policy(policy_path_from_env())
    except PolicyError as exc:
        print(f"library-keeper: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    serve(build_server(policy, audit_from_env()))


if __name__ == "__main__":
    main()
