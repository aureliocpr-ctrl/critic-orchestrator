"""Provider backends: *how* a single adversarial worker actually runs.

The orchestrator is provider-agnostic. A backend takes a `WorkerSpec`
(prompt + JSON schema) and returns a `BackendResult` (the parsed verdict,
or an error). Two families exist, with a hard, honest capability split:

  * AGENTIC — a coding-agent CLI that owns real Bash/Read/Grep tools and
    can therefore run EVERY worker, including `falsification`
    (`git stash` + `pytest`) and `caller_verification` (`grep` the call
    sites). The reference agentic backend is the built-in Claude CLI path
    in `orchestrator._spawn_worker` (kept there so the existing test suite,
    which patches `orchestrator.subprocess.Popen`, stays green). Other
    agent CLIs (Codex, etc.) plug in via a future `CommandBackend` — see
    NOTE at the bottom.

  * REASONING — a plain chat-completion API (Anthropic native, or any
    OpenAI-compatible endpoint: OpenAI, DeepSeek, Kimi/Moonshot, Together,
    OpenRouter, Groq, local vLLM/Ollama). ONE request, JSON-constrained.
    It CANNOT execute git/pytest/grep, so a worker whose
    `requires_execution` flag is set is reported as
    `skipped: requires an agentic backend` — never fabricated. The
    `counterexample` worker (pure reasoning over the supplied claim/diff)
    runs normally; its quality on an API backend depends on how much
    context the caller puts in `diff_summary`, since the model has no file
    access.

PROVIDERS DIFFER — structured output is not uniform. Two shapes verified
live 2026-07-01:
  * Moonshot kimi-k2.7-code accepts `response_format:{type:json_schema}`
    but rejects any temperature != 1 (HTTP 400).
  * DeepSeek (deepseek-v4-*) rejects `response_format:{type:json_schema}`
    ("this response_format type is unavailable") — it needs
    `{type:json_object}` (or none) plus the schema described in the prompt.
So the OpenAI-compatible backend tries, in order, `json_schema` →
`json_object` → no response_format (`CRITIC_JSON_MODE=auto`, the default),
falling back only on the specific "response_format unsupported" 400, and
never sends a temperature unless `CRITIC_TEMPERATURE` is set.

A third family bridges the gap on Windows:

  * GHOST-CLI (`ghost_cli`, see ghost_backend.py) — a fresh INTERACTIVE
    hidden Claude session per worker, driven via `clp ai-eye`. Agentic
    (runs execution workers) but WITHOUT `claude --print`, so it stays on
    the flat subscription when headless calls become metered.

Selection via env (`make_backend_from_env`):
    CRITIC_BACKEND = claude_cli (default) | ghost_cli | anthropic_api | openai_compat
    CRITIC_MODEL   = model id (required for the API backends)
    CRITIC_JSON_MODE = auto (default) | json_schema | json_object | none
    CRITIC_TEMPERATURE = optional float (omitted unless set)
    ANTHROPIC_API_KEY / CRITIC_API_KEY (or OPENAI_API_KEY) / CRITIC_BASE_URL

`make_backend_from_env` returns `None` for the default `claude_cli` case.
No third-party HTTP dependency: the API backends use `urllib`.

HONEST TEST STATUS: the API backends are unit-tested (mocked `urllib`) and
live smoke-tested against Moonshot Kimi (json_schema) and DeepSeek
(json_object fallback). They are not part of the CI run (need a real key).
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .orchestrator import WorkerSpec


@dataclass
class BackendResult:
    """Outcome of running one worker through a backend."""

    verdict: dict[str, Any] | None
    error: str | None
    cost_usd: float = 0.0
    raw_preview: str = ""


_SKIP_MSG = (
    "skipped: this worker requires an agentic backend with tool access "
    "(Bash/Grep for git stash / pytest / call-site search); the current "
    "API backend can only run reasoning-only workers"
)


def _post_json(
    url: str, headers: dict[str, str], body: dict[str, Any], timeout: int,
) -> dict[str, Any]:
    """POST a JSON body and return the parsed JSON response. Raises
    urllib.error.HTTPError / URLError on transport failure."""
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={**headers, "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


def _http_detail(exc: urllib.error.HTTPError) -> str:
    try:
        return exc.read().decode("utf-8", "replace")[:300]
    except Exception:
        return str(exc)


class OpenAICompatibleBackend:
    """Any OpenAI-compatible /chat/completions endpoint (OpenAI, DeepSeek,
    Kimi/Moonshot, Together, OpenRouter, Groq, local vLLM/Ollama).

    Structured output is negotiated per provider: `json_mode='auto'` tries
    `json_schema`, then `json_object`, then no response_format, falling
    back only on the specific "response_format unsupported" 400.
    """

    supports_execution = False

    def __init__(self, *, base_url: str, api_key: str, model: str,
                 temperature: float | None = None,
                 json_mode: str = "auto") -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        # Temperature is NOT sent unless explicitly set: some reasoning
        # models reject temperature != 1 (e.g. Moonshot kimi-k2.7-code
        # returns HTTP 400). Omitting it lets each provider use its default.
        self.temperature = temperature
        self.json_mode = json_mode
        self.name = f"openai_compat:{model}"

    def _format_order(self, spec: WorkerSpec) -> list[dict[str, Any] | None]:
        schema_fmt: dict[str, Any] = {
            "type": "json_schema",
            "json_schema": {"name": "verdict", "strict": True,
                            "schema": spec.schema},
        }
        object_fmt: dict[str, Any] = {"type": "json_object"}
        table: dict[str, list[dict[str, Any] | None]] = {
            "json_schema": [schema_fmt],
            "json_object": [object_fmt],
            "none": [None],
            "auto": [schema_fmt, object_fmt, None],
        }
        return table.get(self.json_mode, table["auto"])

    def run_worker(
        self, spec: WorkerSpec, project_dir: Path, timeout: int,
    ) -> BackendResult:
        if spec.requires_execution:
            return BackendResult(verdict=None, error=_SKIP_MSG)
        headers = {"Authorization": f"Bearer {self.api_key}"}
        order = self._format_order(spec)
        last_err: BackendResult | None = None
        for i, rf in enumerate(order):
            body: dict[str, Any] = {
                "model": self.model,
                "messages": [{"role": "user", "content": spec.prompt}],
            }
            if rf is not None:
                body["response_format"] = rf
            if self.temperature is not None:
                body["temperature"] = self.temperature
            try:
                resp = _post_json(self.base_url + "/chat/completions",
                                  headers, body, timeout)
            except urllib.error.HTTPError as exc:
                detail = _http_detail(exc)
                # Fall back ONLY when the provider rejects the
                # response_format itself and we have another shape to try.
                more = i < len(order) - 1
                if rf is not None and more and "response_format" in detail.lower():
                    last_err = BackendResult(None, f"http {exc.code}: {detail}")
                    continue
                return BackendResult(None, f"http {exc.code}: {detail}")
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                return BackendResult(None, f"transport error: {exc!r}")
            try:
                content = resp["choices"][0]["message"]["content"]
                verdict = json.loads(content)
            except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
                return BackendResult(None, f"unparseable response: {exc!r}",
                                     raw_preview=json.dumps(resp)[:500])
            if not isinstance(verdict, dict):
                return BackendResult(None, "response was not a JSON object")
            usage = resp.get("usage", {})
            fmt = (rf.get("type") if rf else "none")
            return BackendResult(verdict, None, 0.0,
                                 f"format={fmt} tokens={usage}")
        return last_err or BackendResult(None, "all response_format attempts failed")


class AnthropicAPIBackend:
    """Anthropic native Messages API. Structured output via a forced
    single-tool call (`tool_choice` pinned to `emit_verdict`)."""

    supports_execution = False
    _ENDPOINT = "https://api.anthropic.com/v1/messages"
    _VERSION = "2023-06-01"

    def __init__(self, *, api_key: str, model: str,
                 max_tokens: int = 2048) -> None:
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.name = f"anthropic_api:{model}"

    def run_worker(
        self, spec: WorkerSpec, project_dir: Path, timeout: int,
    ) -> BackendResult:
        if spec.requires_execution:
            return BackendResult(verdict=None, error=_SKIP_MSG)
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": spec.prompt}],
            "tools": [{
                "name": "emit_verdict",
                "description": "Emit the structured adversarial verdict.",
                "input_schema": spec.schema,
            }],
            "tool_choice": {"type": "tool", "name": "emit_verdict"},
        }
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self._VERSION,
        }
        try:
            resp = _post_json(self._ENDPOINT, headers, body, timeout)
        except urllib.error.HTTPError as exc:
            return BackendResult(None, f"http {exc.code}: {_http_detail(exc)}")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            return BackendResult(None, f"transport error: {exc!r}")
        verdict = None
        for block in resp.get("content", []) or []:
            if isinstance(block, dict) and block.get("type") == "tool_use" \
                    and block.get("name") == "emit_verdict":
                verdict = block.get("input")
                break
        if not isinstance(verdict, dict):
            return BackendResult(None, "no emit_verdict tool_use in response",
                                 raw_preview=json.dumps(resp)[:500])
        usage = resp.get("usage", {})
        return BackendResult(verdict, None, 0.0, f"tokens={usage}")


_DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5"


def make_backend_from_env() -> Any | None:
    """Build the backend selected by CRITIC_BACKEND.

    Returns `None` for the default `claude_cli` case (the sentinel that
    tells `adversarial_review` to use its built-in Claude CLI path). Raises
    `ValueError` for a misconfigured API backend so the MCP server can
    surface a clear error instead of failing every worker.
    """
    kind = (os.environ.get("CRITIC_BACKEND") or "claude_cli").strip().lower()
    if kind in ("", "claude_cli", "cli", "claude"):
        return None
    if kind in ("ghost_cli", "ghost"):
        # Lazy import: ghost_backend imports BackendResult from here.
        from .ghost_backend import make_ghost_backend_from_env
        return make_ghost_backend_from_env()
    if kind in ("anthropic_api", "anthropic"):
        key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not key:
            raise ValueError("CRITIC_BACKEND=anthropic_api requires ANTHROPIC_API_KEY")
        model = (os.environ.get("CRITIC_MODEL") or _DEFAULT_ANTHROPIC_MODEL).strip()
        return AnthropicAPIBackend(api_key=key, model=model)
    if kind in ("openai_compat", "openai", "deepseek", "kimi", "moonshot",
                "openrouter", "together", "groq", "local"):
        key = (os.environ.get("CRITIC_API_KEY")
               or os.environ.get("OPENAI_API_KEY") or "").strip()
        base = (os.environ.get("CRITIC_BASE_URL")
                or os.environ.get("OPENAI_BASE_URL") or "").strip()
        model = (os.environ.get("CRITIC_MODEL") or "").strip()
        if not base:
            raise ValueError(
                "CRITIC_BACKEND=openai_compat requires CRITIC_BASE_URL "
                "(e.g. https://api.deepseek.com)")
        if not model:
            raise ValueError("CRITIC_BACKEND=openai_compat requires CRITIC_MODEL")
        temp_raw = (os.environ.get("CRITIC_TEMPERATURE") or "").strip()
        temperature = float(temp_raw) if temp_raw else None
        json_mode = (os.environ.get("CRITIC_JSON_MODE") or "auto").strip().lower()
        # A missing key is allowed for keyless local servers (Ollama).
        return OpenAICompatibleBackend(base_url=base, api_key=key, model=model,
                                       temperature=temperature, json_mode=json_mode)
    raise ValueError(f"unknown CRITIC_BACKEND: {kind!r}")


__all__ = [
    "BackendResult",
    "OpenAICompatibleBackend",
    "AnthropicAPIBackend",
    "make_backend_from_env",
]

# NOTE — pluggable agent CLIs (Codex, Cursor CLI, etc.)
# The built-in Claude CLI backend lives in orchestrator._spawn_worker; the
# first alternative AGENTIC backend is ghost_cli (ghost_backend.py, hidden
# interactive Claude sisters — Windows-only). A generic `CommandBackend`
# (argv template + stdout JSON extractor) remains the planned way to add
# other agent CLIs. It is deliberately NOT shipped yet: each agent exposes
# structured JSON output through different flags, and inventing those flags
# unverified would be exactly the confabulation this tool exists to prevent.
