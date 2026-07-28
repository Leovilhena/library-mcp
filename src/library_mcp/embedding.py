"""Local Ollama embedding client.

The base URL comes only from policy, set once at server startup -- nothing
in this module accepts a URL from a tool call or from document content.
There is no "fetch this address" capability anywhere in this server, so the
SSRF concern sandbox-fetch has to actively defend against (an
attacker-influenceable destination) doesn't apply here: every outbound call
this process ever makes goes to the one address the operator configured.

`keep_alive` is set short so the embedding model doesn't stay resident and
compete with the chat model for VRAM -- same reasoning as fetch.yaml's
injection_review model.
"""

from __future__ import annotations

import httpx


class EmbeddingError(Exception):
    """The embedding call failed."""


class EmbeddingClient:
    def __init__(self, base_url: str, model: str, timeout_seconds: float) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout_seconds

    async def embed(self, text: str) -> list[float]:
        """Async so a long ingestion (hundreds of sequential calls) doesn't
        block the event loop for the whole server -- a synchronous client
        here meant `ask_library` (and every other request) stalled for the
        entire duration of any `learn` call in progress. Found live
        2026-07-28: a 14 MB book took long enough to embed that it looked
        like the tool had hung.
        """
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"{self._base_url}/api/embeddings",
                    json={"model": self._model, "prompt": text, "keep_alive": "30s"},
                )
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPError as exc:
            msg = f"embedding request failed: {exc}"
            raise EmbeddingError(msg) from exc
        embedding = data.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            msg = "embedding response missing a non-empty 'embedding' list"
            raise EmbeddingError(msg)
        return [float(x) for x in embedding]
