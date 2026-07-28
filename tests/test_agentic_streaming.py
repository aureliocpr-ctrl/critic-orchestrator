"""Tests for streaming in the agentic backend.

WHY STREAMING IS NOT A NICE-TO-HAVE HERE
A non-streaming request stays silent for as long as the model thinks. A
reasoning model on a review prompt thinks for a minute or more, and a
silent connection is what intermediaries kill: Kimi k3 died with
ConnectionResetError at 145 s twice, and I wrote it off as "the provider
is flaky". Diagnosis said otherwise — the provider answers a long
tool-calling request fine (50.6 s) and streams the same prompt with the
first byte at 1.8 s, and its documented server timeout is two hours. The
fault was mine: no streaming. Moonshot's own docs say streaming is
"strongly recommended" for long tasks precisely because of proxy
buffering and read timeouts.

THE DELICATE PART is reassembly. In streaming, tool calls arrive as
partial deltas keyed by `index`: the id and name land in one chunk, the
arguments dribble across many, and a second tool call interleaves under
its own index. Rebuilding them wrong yields either a dropped tool call or
JSON that will not parse — both indistinguishable from a model failure,
which is why they are pinned here rather than trusted.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from critic_orchestrator.agentic_api import (
    AgenticApiBackend,
    _assemble_stream,
)
from critic_orchestrator.orchestrator import WorkerSpec

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "claim_holds": {"type": "boolean"},
        "evidence": {"type": "string"},
    },
    "required": ["claim_holds", "evidence"],
}


def _spec() -> WorkerSpec:
    return WorkerSpec(name="premortem", prompt="review it", schema=_SCHEMA)


def _sse(*chunks: dict) -> list[bytes]:
    """Render deltas as the wire bytes an SSE endpoint sends."""
    out = [b"data: " + json.dumps(c).encode() + b"\n\n" for c in chunks]
    out.append(b"data: [DONE]\n\n")
    return out


def _delta(**delta: Any) -> dict:
    return {"choices": [{"index": 0, "delta": delta}]}


class _StreamResponse:
    """Minimal file-like SSE response: iterating yields wire lines."""

    def __init__(self, lines: list[bytes]) -> None:
        self._lines = list(lines)

    def __iter__(self):
        return iter(self._lines)

    def __enter__(self) -> _StreamResponse:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self) -> bytes:  # pragma: no cover - streaming path only
        return b"".join(self._lines)


# ---------------------------------------------------------------------------
# Reassembly
# ---------------------------------------------------------------------------

def test_content_deltas_are_concatenated() -> None:
    msg = _assemble_stream(_StreamResponse(_sse(
        _delta(role="assistant", content="Hello"),
        _delta(content=" there"),
        _delta(content="!"),
    )))
    assert msg["content"] == "Hello there!"
    assert msg.get("tool_calls") in (None, [])


def test_a_tool_call_split_across_chunks_is_rebuilt() -> None:
    """id+name in one chunk, arguments dribbled across three."""
    msg = _assemble_stream(_StreamResponse(_sse(
        _delta(role="assistant", tool_calls=[{
            "index": 0, "id": "call_a", "type": "function",
            "function": {"name": "fs_read", "arguments": ""},
        }]),
        _delta(tool_calls=[{"index": 0,
                            "function": {"arguments": '{"pa'}}]),
        _delta(tool_calls=[{"index": 0,
                            "function": {"arguments": 'th": "m.'}}]),
        _delta(tool_calls=[{"index": 0, "function": {"arguments": 'py"}'}}]),
    )))
    assert len(msg["tool_calls"]) == 1
    call = msg["tool_calls"][0]
    assert call["id"] == "call_a"
    assert call["function"]["name"] == "fs_read"
    assert json.loads(call["function"]["arguments"]) == {"path": "m.py"}


def test_two_interleaved_tool_calls_stay_separate() -> None:
    """Parallel calls interleave under their own index — a reassembly that
    ignores index would concatenate them into one unparseable blob."""
    msg = _assemble_stream(_StreamResponse(_sse(
        _delta(role="assistant", tool_calls=[
            {"index": 0, "id": "a", "type": "function",
             "function": {"name": "fs_read", "arguments": ""}},
            {"index": 1, "id": "b", "type": "function",
             "function": {"name": "fs_grep", "arguments": ""}},
        ]),
        _delta(tool_calls=[{"index": 1,
                            "function": {"arguments": '{"pattern":'}}]),
        _delta(tool_calls=[{"index": 0,
                            "function": {"arguments": '{"path":"m.py"}'}}]),
        _delta(tool_calls=[{"index": 1, "function": {"arguments": '"TODO"}'}}]),
    )))
    calls = {c["function"]["name"]: c for c in msg["tool_calls"]}
    assert set(calls) == {"fs_read", "fs_grep"}
    assert json.loads(calls["fs_read"]["function"]["arguments"]) == {
        "path": "m.py"}
    assert json.loads(calls["fs_grep"]["function"]["arguments"]) == {
        "pattern": "TODO"}
    # Order follows index, not arrival.
    assert [c["id"] for c in msg["tool_calls"]] == ["a", "b"]


def test_done_sentinel_and_blank_lines_are_ignored() -> None:
    msg = _assemble_stream(_StreamResponse([
        b"\n",
        b": keep-alive comment\n",
        b"data: " + json.dumps(_delta(content="x")).encode() + b"\n\n",
        b"data: [DONE]\n\n",
        b"data: {\"should\": \"never be read\"}\n\n",
    ]))
    assert msg["content"] == "x"


def test_malformed_chunk_does_not_abort_the_stream() -> None:
    """One unparseable chunk must not lose the whole answer."""
    msg = _assemble_stream(_StreamResponse([
        b"data: {not json\n\n",
        b"data: " + json.dumps(_delta(content="survived")).encode() + b"\n\n",
        b"data: [DONE]\n\n",
    ]))
    assert msg["content"] == "survived"


def test_reasoning_deltas_do_not_pollute_the_answer() -> None:
    """Thinking models emit reasoning_content separately. It must not be
    concatenated into content, or a verdict parsed from prose would pick
    up the model's scratchpad."""
    msg = _assemble_stream(_StreamResponse(_sse(
        _delta(role="assistant", reasoning_content="let me think..."),
        _delta(reasoning_content=" more thinking"),
        _delta(content='{"claim_holds": true, "evidence": "e"}'),
    )))
    assert msg["content"] == '{"claim_holds": true, "evidence": "e"}'
    assert "think" not in msg["content"]


def test_empty_stream_yields_an_empty_message() -> None:
    msg = _assemble_stream(_StreamResponse([b"data: [DONE]\n\n"]))
    assert msg["content"] == ""
    assert not msg.get("tool_calls")


# ---------------------------------------------------------------------------
# Loop integration
# ---------------------------------------------------------------------------

def _backend(**kw: Any) -> AgenticApiBackend:
    defaults = {"base_url": "https://api.example.com", "api_key": "k",
                    "model": "kimi-k3", "max_steps": 6, "stream": True,
                    "require_investigation": False}
    defaults.update(kw)
    return AgenticApiBackend(**defaults)  # type: ignore[arg-type]


def test_streaming_is_requested_when_enabled(tmp_path: Path) -> None:
    seen: list[dict] = []

    def _fake(req: Any, timeout: float | None = None) -> _StreamResponse:
        seen.append(json.loads(req.data.decode()))
        return _StreamResponse(_sse(_delta(role="assistant", tool_calls=[{
            "index": 0, "id": "c1", "type": "function",
            "function": {"name": "submit_verdict",
                         "arguments": json.dumps(
                             {"claim_holds": True, "evidence": "e"})},
        }])))

    with patch("critic_orchestrator.agentic_api.urllib.request.urlopen",
               _fake):
        res = _backend().run_worker(_spec(), tmp_path, 60)
    assert res.verdict == {"claim_holds": True, "evidence": "e"}
    assert seen[0]["stream"] is True


def test_streaming_off_by_default_keeps_the_json_path(tmp_path: Path) -> None:
    """Default stays non-streaming: it is the shape verified against three
    providers. Streaming is opt-in per backend / CRITIC_STREAM."""
    b = AgenticApiBackend(base_url="https://x/v1", api_key="k", model="m")
    assert b.stream is False


def test_a_streamed_tool_call_drives_the_loop(tmp_path: Path) -> None:
    """End to end: streamed fs_read, then a streamed verdict."""
    (tmp_path / "m.py").write_text("x = 1\n")
    responses = [
        _sse(_delta(role="assistant", tool_calls=[{
            "index": 0, "id": "c1", "type": "function",
            "function": {"name": "fs_read", "arguments": ""}}]),
             _delta(tool_calls=[{"index": 0, "function": {
                 "arguments": '{"path": "m.py"}'}}])),
        _sse(_delta(role="assistant", tool_calls=[{
            "index": 0, "id": "c2", "type": "function",
            "function": {"name": "submit_verdict",
                         "arguments": json.dumps(
                             {"claim_holds": False, "evidence": "line 1"})}}])),
    ]
    calls: list[dict] = []

    def _fake(req: Any, timeout: float | None = None) -> _StreamResponse:
        calls.append(json.loads(req.data.decode()))
        return _StreamResponse(responses[len(calls) - 1])

    with patch("critic_orchestrator.agentic_api.urllib.request.urlopen",
               _fake):
        res = _backend().run_worker(_spec(), tmp_path, 60)
    assert res.verdict == {"claim_holds": False, "evidence": "line 1"}
    tool_msgs = [m for m in calls[1]["messages"] if m["role"] == "tool"]
    assert tool_msgs and "x = 1" in tool_msgs[0]["content"]


def test_env_enables_streaming(monkeypatch: pytest.MonkeyPatch) -> None:
    from critic_orchestrator.backends import make_backend_from_env
    monkeypatch.setenv("CRITIC_BACKEND", "agentic_api")
    monkeypatch.setenv("CRITIC_MODEL", "kimi-k3")
    monkeypatch.setenv("CRITIC_API_KEY", "k")
    monkeypatch.setenv("CRITIC_BASE_URL", "https://api.moonshot.ai/v1")
    monkeypatch.setenv("CRITIC_STREAM", "1")
    b = make_backend_from_env()
    assert b.stream is True
