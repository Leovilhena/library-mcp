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

from mcp.server.fastmcp import FastMCP

from library_mcp.audit import AuditLog, Event
from library_mcp.config import KeeperPolicy, PolicyError, load_keeper_policy, policy_path_from_env
from library_mcp.embedding import EmbeddingClient, EmbeddingError
from library_mcp.keeper_model import AnswerDecision, ReasoningClient, ReasoningError, SearchDecision
from library_mcp.runtime import audit_from_env, build_app, serve
from library_mcp.store import SearchResult, open_store, search


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


def _ask(deps: _Deps, question: str) -> str:
    policy = deps.policy
    deps.audit.write(Event.ASKED, detail=question)

    conn = open_store(policy.db_path)
    embedder = EmbeddingClient(policy.ollama_base_url, policy.embedding_model, policy.embed_timeout_seconds)
    reasoner = ReasoningClient(policy.ollama_base_url, policy.reasoning_model, policy.reasoning_timeout_seconds)

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
            query_vector = embedder.embed(query)
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
            decision = reasoner.decide(query, context, remaining)
        except ReasoningError as exc:
            deps.audit.write(Event.ANSWER_FAILED, detail=question, reason=f"reasoning failed: {exc}")
            return "I couldn't reason over the retrieved passages right now."

        if isinstance(decision, AnswerDecision):
            deps.audit.write(Event.ANSWERED, detail=question, searches=attempt + 1)
            return decision.text
        if isinstance(decision, SearchDecision):
            query = decision.query
            continue

    # Exhausted max_searches without an explicit answer -- return whatever
    # context was gathered rather than silently giving up.
    deps.audit.write(Event.ANSWER_FAILED, detail=question, reason="max_searches exhausted")
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
    def ask_library(question: str) -> str:
        return _ask(deps, question)

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
