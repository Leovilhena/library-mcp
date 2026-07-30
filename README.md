<div align="center">

# library-mcp

A specialized "library keeper" sub-agent for the Pythia stack: ingest PDF/EPUB
books, then answer questions grounded in them via local embedding search.

[![CI](https://github.com/Leovilhena/library-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/Leovilhena/library-mcp/actions/workflows/ci.yml)
[![Images](https://github.com/Leovilhena/library-mcp/actions/workflows/images.yml/badge.svg)](https://github.com/Leovilhena/library-mcp/actions/workflows/images.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Checked with mypy](https://img.shields.io/badge/mypy-strict-2a6db2)](pyproject.toml)
[![Linted with ruff](https://img.shields.io/badge/ruff-passing-261230)](pyproject.toml)

</div>

Two MCP servers, one shared knowledge base:

- **`library-parse`**:
  - `learn(file_name)` extracts, chunks, and embeds a book from a shared
    inbox/attachment-cache directory into `knowledge.db`, classifying it
    as a book/article/notes deterministically (structural markers, no
    LLM) and extracting a real title from file metadata when present.
  - `learn_text(title, text, source_url)` ingests already-extracted text
    (e.g. a fetched web page), the same way, minus the file-parsing step.
  - `forget(title)` removes a learned book by title or partial title,
    refusing on zero or multiple matches rather than guessing.
  - `list_learned()` lists everything in the store, including in-progress
    embedding jobs, with each book's doc_type.
- **`library-keeper`**:
  - `ask_library(question)` embeds the question, searches the shared
    store, and hands matches to a small local reasoning model that decides
    whether to answer (with citations) or search again for connected
    knowledge, bounded at a few attempts. The frontier model calls this
    once and gets one synthesized answer back. Deterministically logs a
    "knowledge gap" (never from parsing the model's own wording) when a
    question can't be confidently answered.
  - `list_knowledge_gaps()` lists open gaps — questions worth adding a
    source for — for human review.

Both servers always talk to the same fixed local address for embeddings and
reasoning — that address is set once, in a config file, and can't be changed
by a book's content or by anything a user types. So there's nothing in a PDF
or a chat message that could redirect this to some other server.

More on how this was actually built and debugged — including the real bugs
found along the way (a small reasoning model not following instructions
reliably, an EPUB's table of contents accidentally confusing search results,
Project Gutenberg's own licence text getting ingested and summarized as if it
were the book, a tricky container-permissions issue) — is written up in the Pythia stack's
own docs: `hermes-stack/docs/architecture/library-mcp.md`.

## Development

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
```

## License

MIT.
