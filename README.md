# library-mcp

A specialized "library keeper" sub-agent for the Pythia stack: ingest PDF/EPUB
books, then answer questions grounded in them via local embedding search.

Two MCP servers, one shared knowledge base:

- **`library-parse`** — `learn(file_name)` extracts, chunks, and embeds a
  book from a shared inbox directory into `knowledge.db`. `list_learned()`
  lists what's in there.
- **`library-keeper`** — `ask_library(question)` embeds the question,
  searches the shared store, and hands matches to a small local reasoning
  model that decides whether to answer (with citations) or search again for
  connected knowledge, bounded at a few attempts. The frontier model calls
  this once and gets one synthesized answer back.

Every embedding/reasoning call goes to one address set in policy at
startup — never a tool argument, never document content — so there's no
attacker-influenceable fetch target anywhere in this project, unlike a
general-purpose web fetcher.

Full design, threat model, and the real findings from building this
(a reasoning model's format-compliance failure, a retrieval-quality bug from
EPUB table-of-contents noise, a `podman --userns keep-id` permissions
gotcha) are in the Pythia stack's own docs:
`hermes-stack/docs/architecture/library-mcp.md` and
`hermes-stack/docs/decisions/0004-*.md` / `0005-*.md`.

## Development

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
```

## License

MIT.
