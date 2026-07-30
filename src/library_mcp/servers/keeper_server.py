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
from typing import Literal

from mcp.server.fastmcp import FastMCP

from library_mcp.audit import AuditLog, Event
from library_mcp.config import KeeperPolicy, PolicyError, load_keeper_policy, policy_path_from_env
from library_mcp.embedding import EmbeddingClient, EmbeddingError
from library_mcp.keeper_model import AnswerDecision, ReasoningClient, ReasoningError, SearchDecision
from library_mcp.runtime import audit_from_env, build_app, serve
from library_mcp.store import (
    GlossaryEntry,
    SearchResult,
    find_book_by_filename,
    lookup_glossary_term,
    mark_followups_delivered,
    open_store,
    record_knowledge_gap,
    search,
)
from library_mcp.store import list_knowledge_gaps as list_knowledge_gaps_fn
from library_mcp.store import (
    list_pending_followups as list_pending_followups_fn,
)

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

# Book Q&A escalation flow (docs/planning/book-qa-escalation-flow.md §1):
# deterministic query cleanup, run first inside `_ask()`, before anything
# else touches `question`. Named, not guessed, same discipline
# `_TERM_LOOKUP_RE` above already models -- each entry below is a known,
# fixed string with a known origin, not a fuzzy "looks like noise" guess.
#
# Currently just quiet mode's directive block
# (`~/.pythia/plugins/quiet_mode/__init__.py`'s `_DIRECTIVE` constant,
# matched here verbatim): when quiet mode is on, this exact text is
# prepended to the user's raw message before the frontier model ever sees
# it, which means it can reach `ask_library`'s `question` argument
# unparaphrased if the model paraphrases lazily or its tool-argument
# construction (itself only 40-60% reliable) leaks it through verbatim.
# Add any future plugin-injected prefix to this same list as it's
# discovered -- do not guess at one speculatively.
_WRAPPER_PREFIXES: tuple[str, ...] = (
    # Current wording (2026-07-30, reworded same day the "brief" phrasing
    # was found causing the model to truncate requested content, not just
    # trim style -- see quiet_mode's own __init__.py for the incident).
    "[Quiet mode is ON. Answer only the message below. Keep your reply free "
    "of greetings, small talk, and unnecessary preamble -- get straight to "
    "the point. This means trimming STYLE, not CONTENT: if the user asked "
    "for a full list or a complete answer, give the whole thing plainly "
    "rather than summarizing or truncating it for the sake of brevity. Do "
    "not ask a follow-up question. Do not bring up earlier topics or add "
    "commentary beyond what was asked.]",
    # Prior wording, kept so already-in-flight conversation history (a turn
    # sent before this rollout, or an older cached agent instance not yet
    # restarted) still strips cleanly instead of leaking through unmatched.
    "[Quiet mode is ON. Answer only the message below, as briefly as "
    "possible. Do not ask a follow-up question. Do not bring up earlier "
    "topics or add commentary beyond what was asked.]",
)

# Deterministic-only per the design doc's §1.1 decision: no LLM-based
# rewrite. Whitespace/newline collapse plus a small set of leading/trailing
# quote and punctuation artifacts to trim -- never touches the words of the
# actual question, only wrapper text and formatting noise around it.
_WHITESPACE_RE = re.compile(r"\s+")
_QUOTE_STRIP_CHARS = "\"'“”‘’`"  # noqa: RUF001 -- real curly quotes users actually type, not a typo


def _clean_question(question: str) -> str:
    """Deterministic Stage-1 cleanup (§1 of the design doc): strip known
    wrapper-text prefixes, collapse whitespace/newlines to single spaces,
    trim surrounding quote/punctuation artifacts. No model call -- cannot
    make a query *worse*, since it only removes text that is never part of
    the user's real question. Called as the very first line of `_ask()`,
    before `_term_lookup_candidate()` and before the first embed call, so
    the cleanup applies to the exact string that drives retrieval."""
    cleaned = question
    for prefix in _WRAPPER_PREFIXES:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :]
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
    return cleaned.strip(_QUOTE_STRIP_CHARS).strip()


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


@dataclass(frozen=True, slots=True)
class AskOutcome:
    """Book Q&A escalation flow (docs/planning/book-qa-escalation-flow.md
    §2): `_ask_structured()`'s return value, giving a caller inside this
    same process a machine-readable read on what `_ask()`'s bounded
    embed->search->reason loop actually did, without changing anything
    about `ask_library`'s own `question: str -> str` tool contract (only
    the thin `_ask()` wrapper is exposed to the MCP tool boundary).

    `status` is the existing branches in `_ask()`'s loop, given names --
    not new decision logic. No "answered but low confidence" status: the
    codebase has no existing signal for that today and this design does not
    invent one (§3, §7 non-goals)."""

    text: str
    status: Literal[
        "answered", "no_matches", "repeated_query_no_answer", "embed_failed", "reasoning_failed"
    ]
    gap_id: int | None  # set iff record_knowledge_gap was called this invocation


async def _ask_structured(deps: _Deps, question: str) -> AskOutcome:
    policy = deps.policy
    conn = open_store(policy.db_path)

    # Stage 1 (§1): deterministic query cleanup, first thing, before the
    # term-lookup pre-check and before the first embed call -- see
    # `_clean_question`'s own docstring for why this exact spot.
    cleaned_question = _clean_question(question)

    # Book structuring's deterministic term-lookup pre-check (§4, §9 step
    # 4), run once, ahead of the first search round -- never gates or
    # replaces the real search below, it only adds one extra labeled
    # context block if it fires. `glossary_hit`/`glossary_term` on the
    # ASKED event give step 7's "does the pre-check actually fire on real
    # questions" observation something concrete to check in
    # keeper-audit/audit.jsonl without a new event type.
    term_candidate = _term_lookup_candidate(cleaned_question)
    glossary_entry = lookup_glossary_term(conn, term_candidate) if term_candidate else None
    deps.audit.write(
        Event.ASKED,
        detail=question,
        cleaned_query=cleaned_question,
        glossary_hit=glossary_entry is not None,
        glossary_term=glossary_entry.term if glossary_entry else None,
    )

    embedder = EmbeddingClient(
        policy.ollama_base_url, policy.embedding_model, policy.embed_timeout_seconds
    )
    reasoner = ReasoningClient(
        policy.ollama_base_url, policy.reasoning_model, policy.reasoning_timeout_seconds
    )

    query = cleaned_question
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
            return AskOutcome(
                text="I couldn't search the library right now (embedding call failed).",
                status="embed_failed",
                gap_id=None,
            )

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
            return AskOutcome(
                text="I couldn't reason over the retrieved passages right now.",
                status="reasoning_failed",
                gap_id=None,
            )

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
                gap_id = record_knowledge_gap(
                    conn, cleaned_question, "no_matches", datetime.now(UTC).isoformat()
                )
                return AskOutcome(text=decision.text, status="no_matches", gap_id=gap_id)
            return AskOutcome(text=decision.text, status="answered", gap_id=None)
        if isinstance(decision, SearchDecision):
            query = decision.query
            continue

    # Only reachable via the duplicate-query break above -- `_parse_decision`
    # always returns AnswerDecision on the final iteration (remaining == 0),
    # so normal loop exhaustion never falls through to here. Label the
    # reason for what actually happened, not what the code used to assume.
    deps.audit.write(Event.ANSWER_FAILED, detail=question, reason="repeated query, no answer")
    gap_id = record_knowledge_gap(
        conn, cleaned_question, "repeated_query_no_answer", datetime.now(UTC).isoformat()
    )
    fallback_text = (
        "I searched the library but couldn't settle on a confident answer. "
        "Closest passages found:\n\n"
        f"{_format_context(all_results, policy.max_context_chars, glossary_entry)}"
    )
    return AskOutcome(text=fallback_text, status="repeated_query_no_answer", gap_id=gap_id)


# Book Q&A escalation flow (docs/planning/book-qa-escalation-flow.md §4.2):
# honest interim-reply wording for the two "the library genuinely didn't
# settle on an answer" outcomes, in place of `_ask_structured()`'s bare
# fallback text -- adapted from the design doc's own example wordings, one
# per branch, so the live turn's only two possible outcomes are a real
# grounded answer or this kind of honest interim reply. Never used for
# `embed_failed`/`reasoning_failed` (§3): those are infra failures, not
# knowledge gaps, and using this framing there would incorrectly imply "the
# books don't have this" when the real story is "Ollama hiccuped."
_NO_MATCHES_INTERIM_REPLY = (
    "I don't have a confident answer on that from the book right now -- let "
    "me dig deeper and I'll bring it up next time we talk."
)
_REPEATED_QUERY_INTERIM_REPLY = (
    "The library didn't turn up a solid answer to that one on a normal "
    "search. I'm going to do a wider re-search in the background; if it "
    "finds something I'll mention it later."
)


async def _ask(deps: _Deps, question: str) -> str:
    """Thin wrapper preserving `ask_library`'s `question: str -> str` MCP
    contract exactly (§2, §5): the frontier model still just gets an
    answer string back. Applies the Stage 2->3 escalation trigger (§3-4):
    an honest interim reply for the two structurally-detected "no real
    answer" outcomes, and the outcome's own text unchanged for everything
    else (a real answer, or an infra failure that must not be reframed as
    a knowledge gap). Stage 3 itself is not triggered here -- the escalated
    gap is already a normal `knowledge_gaps` row via `_ask_structured()`'s
    unchanged `record_knowledge_gap` call, picked up by the existing
    nightly `gap_research.py` path with zero new code (§7 build plan, step
    6 onward is explicitly out of scope for this build)."""
    outcome = await _ask_structured(deps, question)
    if outcome.status == "no_matches":
        return _NO_MATCHES_INTERIM_REPLY
    if outcome.status == "repeated_query_no_answer":
        return _REPEATED_QUERY_INTERIM_REPLY
    return outcome.text


def build_server(policy: KeeperPolicy, audit: AuditLog) -> FastMCP:
    deps = _Deps(policy=policy, audit=audit)
    app = build_app("library-keeper")

    @app.tool(
        description=(
            "Ask a question grounded in the books that have been learned via the learn tool. "
            "Internally searches the shared knowledge base up to its own configured limit "
            "(possibly more than once, for connected topics across books) BEFORE returning, "
            "and returns one synthesized, cited answer -- broadening the search yourself by "
            "calling this tool again with a rephrased question, in the SAME turn, is redundant "
            "and just doubles latency and failure exposure for no extra coverage. Call it once "
            "per question. A different question later in the conversation -- even about the "
            "same book or topic -- is a NEW question and must get its own fresh call: do not "
            "skip calling it because an earlier turn already covered similar ground, and never "
            "assume a prior attempt's outcome (success, partial result, or failure) still "
            "applies without calling it again. Use this instead of answering from your own "
            "memory when the question is about specific book content."
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

    # Sandbox-vs-ask_library redirect (docs/incidents/2026-07-30-sandbox
    # -vs-ask-library.md, docs/architecture/sandbox-mcp.md): a small,
    # read-only, model-invisible lookup for the gateway's `sandbox_redirect`
    # plugin's `pre_tool_call` hook. Excluded from
    # `mcp_servers.library_keeper.tools` in config.yaml for the same reason
    # `list_pending_followups`/`mark_followup_delivered` are -- this is a
    # deterministic gateway-side check, never something the frontier model
    # should be able to call on its own initiative.

    @app.tool(
        name="find_book_by_filename",
        description=(
            "Internal: look up a book by filename (basename match against "
            "books.source_path, case-insensitive). Used by the gateway's "
            "sandbox_redirect plugin to check whether a shell command's file "
            "argument already refers to a fully-ingested, done-status book "
            "before letting a manual file read run. NOT intended for the "
            "frontier model to call on its own initiative -- excluded from "
            "mcp_servers.library_keeper.tools in config.yaml for exactly "
            "that reason. Use ask_library for actually answering questions."
        ),
    )
    def find_book_by_filename_tool(filename: str) -> str:
        conn = open_store(deps.policy.db_path)
        match = find_book_by_filename(conn, filename)
        deps.audit.write(
            Event.FILENAME_LOOKED_UP,
            detail=filename,
            found=match is not None,
            status=match.status if match else None,
        )
        if match is None:
            return json.dumps({"found": False}, ensure_ascii=False)
        return json.dumps(
            {
                "found": True,
                "title": match.title,
                "source_path": match.source_path,
                "status": match.status,
            },
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
            # The f-string below only builds the placeholder skeleton (a run
            # of literal `?` characters sized from len(delivered), an int)
            # -- never interpolates data. Every real value still goes
            # through parameterized binding, the standard-safe pattern for
            # a variable-length SQLite IN clause.
            rows = conn.execute(
                "SELECT id, gap_id, chat_id FROM pending_followups WHERE id IN "  # noqa: S608  # nosec B608
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
