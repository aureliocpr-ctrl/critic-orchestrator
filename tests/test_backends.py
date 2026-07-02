"""Tests for the provider backends (multi-provider support).

Mocks `urllib` so no real network call or paid token is spent. These
pin request construction + response parsing + the honest execution-skip
behaviour + env-driven selection. They do NOT assert anything about a
live endpoint's behaviour — that needs the user's own key (see
backends.py HONEST TEST STATUS).
"""
from __future__ import annotations

import io
import json
import urllib.error
from pathlib import Path
from unittest.mock import patch

import pytest

from critic_orchestrator import backends
from critic_orchestrator.backends import (
    AnthropicAPIBackend,
    OpenAICompatibleBackend,
    make_backend_from_env,
)
from critic_orchestrator.orchestrator import WorkerSpec

_SCHEMA = {
    "type": "object",
    "properties": {"claim_holds": {"type": "boolean"}},
    "required": ["claim_holds"],
}


def _spec(requires_execution: bool = False) -> WorkerSpec:
    return WorkerSpec(
        name="w", prompt="verify this", schema=_SCHEMA,
        requires_execution=requires_execution,
    )


class _FakeResp:
    """Minimal context-manager stand-in for urlopen's return value."""

    def __init__(self, payload: dict) -> None:
        self._raw = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._raw

    def __enter__(self) -> "_FakeResp":
        return self

    def __exit__(self, *_a) -> bool:
        return False


def _capturing_urlopen(payload: dict, captured: dict):
    def _fake(req, timeout=None):  # noqa: ARG001
        captured["url"] = req.full_url
        captured["headers"] = dict(req.header_items())
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeResp(payload)
    return _fake


# --------------------------------------------------------------------------
# OpenAI-compatible backend
# --------------------------------------------------------------------------

def test_openai_compat_parses_content_and_builds_request(tmp_path: Path) -> None:
    verdict = {"claim_holds": True}
    resp = {"choices": [{"message": {"content": json.dumps(verdict)}}],
            "usage": {"total_tokens": 42}}
    captured: dict = {}
    be = OpenAICompatibleBackend(
        base_url="https://api.deepseek.com/v1", api_key="sk-x", model="deepseek-chat",
    )
    with patch("critic_orchestrator.backends.urllib.request.urlopen",
               _capturing_urlopen(resp, captured)):
        result = be.run_worker(_spec(), tmp_path, timeout=60)

    assert result.error is None
    assert result.verdict == {"claim_holds": True}
    # request shape
    assert captured["url"] == "https://api.deepseek.com/v1/chat/completions"
    assert captured["body"]["model"] == "deepseek-chat"
    assert captured["body"]["response_format"]["type"] == "json_schema"
    # bearer auth (header casing is normalised by urllib to Title-Case)
    auth = {k.lower(): v for k, v in captured["headers"].items()}
    assert auth["authorization"] == "Bearer sk-x"


def test_openai_compat_skips_execution_worker(tmp_path: Path) -> None:
    be = OpenAICompatibleBackend(base_url="https://x/v1", api_key="k", model="m")
    # If it tried to call the network this would raise (no patch installed).
    result = be.run_worker(_spec(requires_execution=True), tmp_path, timeout=60)
    assert result.verdict is None
    assert "agentic backend" in (result.error or "")


def test_openai_compat_omits_temperature_by_default(tmp_path: Path) -> None:
    """Reasoning models (e.g. Moonshot kimi-k2.7-code) return HTTP 400 on
    any temperature != 1. Verified live 2026-07-01. So we must NOT send a
    temperature unless one is explicitly configured."""
    resp = {"choices": [{"message": {"content": json.dumps({"claim_holds": True})}}]}
    captured: dict = {}
    be = OpenAICompatibleBackend(base_url="https://x/v1", api_key="k", model="m")
    with patch("critic_orchestrator.backends.urllib.request.urlopen",
               _capturing_urlopen(resp, captured)):
        be.run_worker(_spec(), tmp_path, timeout=60)
    assert "temperature" not in captured["body"]


def test_openai_compat_sends_temperature_when_set(tmp_path: Path) -> None:
    resp = {"choices": [{"message": {"content": json.dumps({"claim_holds": True})}}]}
    captured: dict = {}
    be = OpenAICompatibleBackend(base_url="https://x/v1", api_key="k",
                                 model="m", temperature=0.0)
    with patch("critic_orchestrator.backends.urllib.request.urlopen",
               _capturing_urlopen(resp, captured)):
        be.run_worker(_spec(), tmp_path, timeout=60)
    assert captured["body"]["temperature"] == 0.0


def test_openai_compat_handles_unparseable(tmp_path: Path) -> None:
    resp = {"choices": [{"message": {"content": "not json {{{"}}]}
    be = OpenAICompatibleBackend(base_url="https://x/v1", api_key="k", model="m")
    with patch("critic_orchestrator.backends.urllib.request.urlopen",
               _capturing_urlopen(resp, {})):
        result = be.run_worker(_spec(), tmp_path, timeout=60)
    assert result.verdict is None
    assert "unparseable" in (result.error or "")


# --------------------------------------------------------------------------
# Anthropic native backend
# --------------------------------------------------------------------------

def test_anthropic_parses_tool_use(tmp_path: Path) -> None:
    verdict = {"claim_holds": False}
    resp = {"content": [
        {"type": "text", "text": "thinking..."},
        {"type": "tool_use", "name": "emit_verdict", "input": verdict},
    ], "usage": {"input_tokens": 10, "output_tokens": 5}}
    captured: dict = {}
    be = AnthropicAPIBackend(api_key="sk-ant-x", model="claude-sonnet-5")
    with patch("critic_orchestrator.backends.urllib.request.urlopen",
               _capturing_urlopen(resp, captured)):
        result = be.run_worker(_spec(), tmp_path, timeout=60)

    assert result.error is None
    assert result.verdict == {"claim_holds": False}
    assert captured["url"] == "https://api.anthropic.com/v1/messages"
    hdr = {k.lower(): v for k, v in captured["headers"].items()}
    assert hdr["x-api-key"] == "sk-ant-x"
    assert hdr["anthropic-version"] == "2023-06-01"
    assert captured["body"]["tool_choice"]["name"] == "emit_verdict"


def test_anthropic_skips_execution_worker(tmp_path: Path) -> None:
    be = AnthropicAPIBackend(api_key="k", model="claude-sonnet-5")
    result = be.run_worker(_spec(requires_execution=True), tmp_path, timeout=60)
    assert result.verdict is None
    assert "agentic backend" in (result.error or "")


def test_anthropic_missing_tool_use_is_error(tmp_path: Path) -> None:
    resp = {"content": [{"type": "text", "text": "no tool call"}]}
    be = AnthropicAPIBackend(api_key="k", model="claude-sonnet-5")
    with patch("critic_orchestrator.backends.urllib.request.urlopen",
               _capturing_urlopen(resp, {})):
        result = be.run_worker(_spec(), tmp_path, timeout=60)
    assert result.verdict is None
    assert "emit_verdict" in (result.error or "")


# --------------------------------------------------------------------------
# Env-driven selection
# --------------------------------------------------------------------------

def test_make_backend_default_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CRITIC_BACKEND", raising=False)
    assert make_backend_from_env() is None


def test_make_backend_claude_cli_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CRITIC_BACKEND", "claude_cli")
    assert make_backend_from_env() is None


def test_make_backend_anthropic_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CRITIC_BACKEND", "anthropic_api")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        make_backend_from_env()


def test_make_backend_anthropic_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CRITIC_BACKEND", "anthropic_api")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    monkeypatch.setenv("CRITIC_MODEL", "claude-opus-4-8")
    be = make_backend_from_env()
    assert isinstance(be, AnthropicAPIBackend)
    assert be.model == "claude-opus-4-8"
    assert be.supports_execution is False


def test_make_backend_openai_requires_base_and_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CRITIC_BACKEND", "openai_compat")
    monkeypatch.delenv("CRITIC_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    with pytest.raises(ValueError, match="CRITIC_BASE_URL"):
        make_backend_from_env()


def test_make_backend_openai_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CRITIC_BACKEND", "deepseek")
    monkeypatch.setenv("CRITIC_BASE_URL", "https://api.deepseek.com/v1")
    monkeypatch.setenv("CRITIC_MODEL", "deepseek-chat")
    monkeypatch.setenv("CRITIC_API_KEY", "sk-x")
    be = make_backend_from_env()
    assert isinstance(be, OpenAICompatibleBackend)
    assert be.model == "deepseek-chat"


def test_make_backend_unknown_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CRITIC_BACKEND", "totally-made-up")
    with pytest.raises(ValueError, match="unknown CRITIC_BACKEND"):
        make_backend_from_env()


# --------------------------------------------------------------------------
# adversarial_review honours a backend (reasoning-only path)
# --------------------------------------------------------------------------

def test_adversarial_review_uses_backend_and_skips_execution(
    tmp_path: Path,
) -> None:
    """With a reasoning-only backend, an execution-required worker is
    skipped (invalid vote) while a reasoning worker votes normally."""
    from critic_orchestrator.orchestrator import adversarial_review

    resp = {"choices": [{"message": {"content": json.dumps({"claim_holds": True})}}]}
    be = OpenAICompatibleBackend(base_url="https://x/v1", api_key="k", model="m")
    reasoning = _spec(requires_execution=False)
    execution = _spec(requires_execution=True)
    execution = WorkerSpec(name="falsify", prompt="p", schema=_SCHEMA,
                           requires_execution=True)

    with patch("critic_orchestrator.backends.urllib.request.urlopen",
               _capturing_urlopen(resp, {})):
        report = adversarial_review(
            claim="c", project_dir=tmp_path,
            workers=[reasoning, execution], timeout=60, backend=be,
        )
    # reasoning worker → hold; execution worker → skipped/invalid
    assert report.votes_hold == 1
    assert report.votes_invalid == 1
    assert report.consensus == "claim_holds"


# --------------------------------------------------------------------------
# Per-provider structured-output negotiation (DeepSeek rejects json_schema)
# --------------------------------------------------------------------------

def test_openai_compat_falls_back_from_json_schema(tmp_path: Path) -> None:
    """DeepSeek returns HTTP 400 'response_format type unavailable' on
    json_schema (verified live 2026-07-01). The backend must fall back to
    json_object and succeed, not give up."""
    ok = {"choices": [{"message": {"content": json.dumps({"claim_holds": True})}}]}
    formats: list = []

    def _fake(req, timeout=None):  # noqa: ARG001
        body = json.loads(req.data.decode("utf-8"))
        rf = body.get("response_format")
        formats.append(rf.get("type") if rf else None)
        if rf and rf.get("type") == "json_schema":
            raise urllib.error.HTTPError(
                req.full_url, 400, "Bad Request", {},
                io.BytesIO(b'{"error":{"message":"This response_format type is unavailable now"}}'),
            )
        return _FakeResp(ok)

    be = OpenAICompatibleBackend(base_url="https://api.deepseek.com",
                                 api_key="k", model="deepseek-v4-pro")
    with patch("critic_orchestrator.backends.urllib.request.urlopen", _fake):
        result = be.run_worker(_spec(), tmp_path, timeout=60)
    assert result.error is None
    assert result.verdict == {"claim_holds": True}
    assert formats == ["json_schema", "json_object"]  # fell back exactly once


def test_openai_compat_non_format_400_does_not_fall_back(tmp_path: Path) -> None:
    """A 400 that is NOT about response_format (e.g. auth) must surface
    immediately, not silently retry with a different format."""
    calls = {"n": 0}

    def _fake(req, timeout=None):  # noqa: ARG001
        calls["n"] += 1
        raise urllib.error.HTTPError(
            req.full_url, 401, "Unauthorized", {},
            io.BytesIO(b'{"error":{"message":"invalid api key"}}'),
        )

    be = OpenAICompatibleBackend(base_url="https://x/v1", api_key="bad", model="m")
    with patch("critic_orchestrator.backends.urllib.request.urlopen", _fake):
        result = be.run_worker(_spec(), tmp_path, timeout=60)
    assert result.verdict is None
    assert "401" in (result.error or "")
    assert calls["n"] == 1  # no wasteful retries


def test_make_backend_openai_json_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CRITIC_BACKEND", "openai_compat")
    monkeypatch.setenv("CRITIC_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("CRITIC_MODEL", "deepseek-v4-pro")
    monkeypatch.setenv("CRITIC_JSON_MODE", "json_object")
    be = make_backend_from_env()
    assert isinstance(be, OpenAICompatibleBackend)
    assert be.json_mode == "json_object"
