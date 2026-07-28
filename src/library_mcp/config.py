"""Policy models and loader.

Policy is trusted input (operator-authored), but still validated strictly --
a malformed policy fails startup rather than degrading to a permissive
default. Same `_Strict` pattern as sandbox-mcp's config.py: unknown keys are
rejected so a typo can't silently disable a control.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class PolicyError(Exception):
    """Policy could not be loaded or is invalid."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ParsePolicy(_Strict):
    # Hardcoded, operator-set destination for embedding calls -- never taken
    # from a tool argument or from ingested document content. There is no
    # "fetch this URL" tool here, so the SSRF concern sandbox-fetch defends
    # against (an attacker-influenceable destination) does not apply: the
    # only outbound call this server ever makes is to this one address.
    ollama_base_url: str
    embedding_model: str = "nomic-embed-text"
    embed_timeout_seconds: Annotated[float, Field(ge=1.0, le=120.0)] = 30.0
    # Where books to ingest are read from. Read-only mount in practice.
    inbox_path: Path
    db_path: Path
    allowed_extensions: list[str] = Field(default_factory=lambda: [".pdf", ".epub"])
    max_file_mb: Annotated[int, Field(ge=1, le=500)] = 100
    chunk_chars: Annotated[int, Field(ge=200, le=20_000)] = 1200
    chunk_overlap_chars: Annotated[int, Field(ge=0, le=5000)] = 150


class KeeperPolicy(_Strict):
    ollama_base_url: str
    embedding_model: str = "nomic-embed-text"
    reasoning_model: str = "gemma3:1b"
    embed_timeout_seconds: Annotated[float, Field(ge=1.0, le=120.0)] = 30.0
    reasoning_timeout_seconds: Annotated[float, Field(ge=1.0, le=300.0)] = 60.0
    db_path: Path
    top_k: Annotated[int, Field(ge=1, le=50)] = 6
    # Bounded internal loop -- mirrors agent.max_iterations already in
    # config.yaml. The keeper may issue one more refined search if the
    # reasoning model decides it needs one, up to this many total searches
    # per question. Small on purpose: an unbounded loop on an unreliable
    # small model is how a single question turns into a stuck agent.
    max_searches: Annotated[int, Field(ge=1, le=5)] = 3
    max_context_chars: Annotated[int, Field(ge=500, le=50_000)] = 6000


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        msg = f"cannot read policy file {path}: {exc}"
        raise PolicyError(msg) from exc
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        msg = f"invalid YAML in {path}: {exc}"
        raise PolicyError(msg) from exc
    if not isinstance(data, dict):
        msg = f"{path} must contain a YAML mapping at the top level"
        raise PolicyError(msg)
    return data


def load_parse_policy(path: Path) -> ParsePolicy:
    try:
        return ParsePolicy(**_load_yaml(path))
    except Exception as exc:  # pydantic.ValidationError, etc.
        msg = f"invalid parse policy {path}: {exc}"
        raise PolicyError(msg) from exc


def load_keeper_policy(path: Path) -> KeeperPolicy:
    try:
        return KeeperPolicy(**_load_yaml(path))
    except Exception as exc:
        msg = f"invalid keeper policy {path}: {exc}"
        raise PolicyError(msg) from exc


def policy_path_from_env(var: str = "POLICY_PATH") -> Path:
    value = os.environ.get(var)
    if not value:
        msg = f"{var} is not set; refusing to start without an explicit policy"
        raise PolicyError(msg)
    return Path(value)
