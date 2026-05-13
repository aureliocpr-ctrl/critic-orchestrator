"""Tests for _spawn_worker after the Popen refactor.

The original implementation used subprocess.run(timeout=...), which
returns only after the worker finishes. That makes external cancellation
impossible: the parent can't reach into a thread blocked inside run().

The refactor switches to Popen + communicate(timeout=...). Before
communicate is awaited, the Popen handle is appended to an optional
`popen_sink` list. A cancel path (from JobRegistry.cancel) iterates the
sink and calls .kill() on each handle.

These tests pin:
  1. Successful path still parses JSON and returns a WorkerVerdict.
  2. The Popen handle is registered in popen_sink BEFORE communicate
     blocks (so an external cancel always finds it).
  3. Timeout still produces a clean WorkerVerdict with error message.
  4. Backwards-compat: popen_sink=None keeps old contract.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from critic_orchestrator.orchestrator import (
    WorkerSpec,
    _spawn_worker,
    adversarial_review,
)


def _make_spec(name: str = "w1") -> WorkerSpec:
    return WorkerSpec(
        name=name,
        prompt="test prompt",
        schema={"type": "object",
                 "properties": {"claim_holds": {"type": "boolean"}},
                 "required": ["claim_holds"]},
        extra_args=("--allowedTools", "Read"),
    )


def _fake_popen_ok(verdict: dict) -> MagicMock:
    """Build a Popen mock that returns a successful claude-CLI envelope."""
    fake = MagicMock(spec=subprocess.Popen)
    payload = json.dumps({
        "is_error": False,
        "total_cost_usd": 0.05,
        "structured_output": verdict,
    })
    fake.communicate.return_value = (payload, "")
    fake.returncode = 0
    fake.poll.return_value = 0
    return fake


def test_spawn_worker_parses_structured_output(tmp_path: Path) -> None:
    spec = _make_spec()
    fake_popen = _fake_popen_ok({"claim_holds": True, "confidence": 0.9,
                                  "evidence": "ok"})
    with patch("critic_orchestrator.orchestrator.subprocess.Popen",
               return_value=fake_popen):
        verdict = _spawn_worker(spec, tmp_path, timeout=60, extra_mcp=None)
    assert verdict.ok
    assert verdict.verdict == {"claim_holds": True, "confidence": 0.9,
                                "evidence": "ok"}
    assert verdict.error is None
    assert verdict.cost_usd == pytest.approx(0.05)


def test_spawn_worker_registers_popen_handle_in_sink(tmp_path: Path) -> None:
    """popen_sink receives the handle BEFORE communicate is awaited.

    This is the cancellability contract: when the spawning thread is
    parked in communicate(), an external thread that holds a reference
    to popen_sink can still call .kill() on the handle.
    """
    spec = _make_spec()
    sink: list[subprocess.Popen] = []

    # The fake's communicate must observe that the handle was appended
    # to the sink before being called. We assert this from inside the
    # side_effect.
    fake_popen = MagicMock(spec=subprocess.Popen)
    fake_popen.poll.return_value = 0
    fake_popen.returncode = 0

    def _side_effect(*_a, **_kw):
        # By the time communicate runs, the sink must already hold us.
        assert fake_popen in sink, "Popen handle not registered before communicate"
        return (json.dumps({
            "is_error": False,
            "total_cost_usd": 0.0,
            "structured_output": {"claim_holds": True, "confidence": 0.9,
                                   "evidence": ""},
        }), "")

    fake_popen.communicate.side_effect = _side_effect

    with patch("critic_orchestrator.orchestrator.subprocess.Popen",
               return_value=fake_popen):
        _spawn_worker(spec, tmp_path, timeout=60, extra_mcp=None,
                      popen_sink=sink)

    assert len(sink) == 1
    assert sink[0] is fake_popen


def test_spawn_worker_timeout_returns_clean_error(tmp_path: Path) -> None:
    spec = _make_spec()
    fake_popen = MagicMock(spec=subprocess.Popen)
    fake_popen.poll.return_value = None
    fake_popen.communicate.side_effect = subprocess.TimeoutExpired(
        cmd="claude", timeout=60,
    )
    with patch("critic_orchestrator.orchestrator.subprocess.Popen",
               return_value=fake_popen):
        verdict = _spawn_worker(spec, tmp_path, timeout=60, extra_mcp=None)
    assert not verdict.ok
    assert "timeout" in (verdict.error or "").lower()
    # After timeout we must have killed the subprocess to free resources.
    fake_popen.kill.assert_called_once()


def test_spawn_worker_handles_missing_claude_binary(tmp_path: Path) -> None:
    spec = _make_spec()
    with patch("critic_orchestrator.orchestrator.subprocess.Popen",
               side_effect=FileNotFoundError("claude not found")):
        verdict = _spawn_worker(spec, tmp_path, timeout=60, extra_mcp=None)
    assert not verdict.ok
    assert verdict.error is not None
    assert "claude" in verdict.error.lower()


def test_spawn_worker_handles_invalid_json_stdout(tmp_path: Path) -> None:
    spec = _make_spec()
    fake_popen = MagicMock(spec=subprocess.Popen)
    fake_popen.poll.return_value = 0
    fake_popen.returncode = 0
    fake_popen.communicate.return_value = ("not-json {{{", "")
    with patch("critic_orchestrator.orchestrator.subprocess.Popen",
               return_value=fake_popen):
        verdict = _spawn_worker(spec, tmp_path, timeout=60, extra_mcp=None)
    assert not verdict.ok
    assert "json_decode" in (verdict.error or "")


def test_spawn_worker_handles_empty_stdout(tmp_path: Path) -> None:
    spec = _make_spec()
    fake_popen = MagicMock(spec=subprocess.Popen)
    fake_popen.poll.return_value = 1
    fake_popen.returncode = 1
    fake_popen.communicate.return_value = ("", "boom")
    with patch("critic_orchestrator.orchestrator.subprocess.Popen",
               return_value=fake_popen):
        verdict = _spawn_worker(spec, tmp_path, timeout=60, extra_mcp=None)
    assert not verdict.ok
    assert "empty stdout" in (verdict.error or "").lower()


def test_spawn_worker_propagates_is_error_payload(tmp_path: Path) -> None:
    """When the claude-CLI itself returns is_error=true, we must surface
    that as a non-ok verdict (not silently treat it as success).
    """
    spec = _make_spec()
    fake_popen = MagicMock(spec=subprocess.Popen)
    fake_popen.poll.return_value = 0
    fake_popen.returncode = 0
    fake_popen.communicate.return_value = (json.dumps({
        "is_error": True,
        "subtype": "error_during_execution",
        "result": "tool fell over",
        "total_cost_usd": 0.01,
    }), "")
    with patch("critic_orchestrator.orchestrator.subprocess.Popen",
               return_value=fake_popen):
        verdict = _spawn_worker(spec, tmp_path, timeout=60, extra_mcp=None)
    assert not verdict.ok
    assert verdict.error is not None
    assert verdict.cost_usd == pytest.approx(0.01)


def test_spawn_worker_backcompat_no_sink(tmp_path: Path) -> None:
    """Calling without popen_sink kwarg must keep the legacy contract."""
    spec = _make_spec()
    fake_popen = _fake_popen_ok({"claim_holds": True, "confidence": 0.5,
                                  "evidence": ""})
    with patch("critic_orchestrator.orchestrator.subprocess.Popen",
               return_value=fake_popen):
        verdict = _spawn_worker(spec, tmp_path, timeout=60, extra_mcp=None)
    assert verdict.ok


def test_adversarial_review_propagates_popen_sink(tmp_path: Path) -> None:
    """adversarial_review must forward popen_sink to every worker so a
    later cancel() can reach all handles.
    """
    spec1 = _make_spec("w1")
    spec2 = _make_spec("w2")
    sink: list[subprocess.Popen] = []

    fake_popen = MagicMock(spec=subprocess.Popen)
    fake_popen.poll.return_value = 0
    fake_popen.returncode = 0
    fake_popen.communicate.return_value = (json.dumps({
        "is_error": False, "total_cost_usd": 0.0,
        "structured_output": {"claim_holds": True, "confidence": 0.5,
                               "evidence": ""},
    }), "")

    with patch("critic_orchestrator.orchestrator.subprocess.Popen",
               return_value=fake_popen):
        report = adversarial_review(
            claim="c", project_dir=tmp_path,
            workers=[spec1, spec2], timeout=60,
            popen_sink=sink,
        )

    assert len(sink) == 2
    assert report.consensus == "claim_holds"
    assert report.votes_hold == 2
