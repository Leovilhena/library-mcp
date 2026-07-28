"""library-keeper MCP server: the specialized sub-agent.

Exposes exactly one tool, `ask_library`, to the frontier model. Everything
between "question in" and "answer out" -- embedding, searching, deciding
whether to search again for connected knowledge, synthesizing a cited answer
-- happens inside this process's own bounded loop, so the frontier's
tool-calling (only 40-60% reliable) never has to orchestrate a multi-turn
exchange itself.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import UTC, datetime

from mcp.server.fastmcp import FastMCP

from library_mcp.audit import AuditLog, Event
from library_mcp.config import KeeperPolicy, PolicyError, load_keeper_policy, policy_path_from_env
from library_mcp.embedding import EmbeddingClient, EmbeddingError
from library_mcp.keeper_model import AnswerDecision, ReasoningClient, ReasoningError, SearchDecision
from library_mcp.runtime import audit_from_env, build_app, serve
from library_mcp.store import SearchResult, open_store, record_knowledge_gap, search
from library_mcp.store import list_knowledge_gaps as list_knowledge_gaps_fn


@dataclass(frozen=True, slots=True)
class _Deps:
    policy: KeeperPolicy
    audit: AuditLog


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


async def _ask(deps: _Deps, question: str) -> str:
    policy = deps.policy
    deps.audit.write(Event.ASKED, detail=question)

    conn = open_store(policy.db_path)
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

        context = _format_context(all_results, policy.max_context_chars)
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
        f"Closest passages found:\n\n{_format_context(all_results, policy.max_context_chars)}"
    )


def build_server(policy: KeeperPolicy, audit: AuditLog) -> FastMCP:
    deps = _Deps(policy=policy, audit=audit)
    app = build_app("library-keeper")

    @app.tool(
        description=(
            "Ask a question grounded in the books that have been learned via the learn tool. "
            "Searches the shared knowledge base (possibly more than once, for connected topics "
            "across books) and returns one synthesized, cited answer. Use this instead of "
            "answering from your own memory when the question is about specific book content."
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
