"""Integration tests for the new async-job MCP tools.

These tests target the four new tools on the MCP server:

    start_adversarial_review   — returns a job_id within <100 ms
    poll_adversarial_review    — returns status + result when done
    cancel_adversarial_review  — kills the in-flight worker subprocesses
    list_adversarial_reviews   — lists known jobs

They call the registered MCP handler directly (no real stdio transport
needed) and patch `subprocess.Popen` so we never actually launch a
`claude` CLI worker — the focus is the async-job orchestration glue,
not the CLI itself.
"""
from __future__ import annotations

import asyncio
import json
import subprocess
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from critic_orchestrator import mcp_server
from critic_orchestrator.job_registry import JobRegistry


@pytest.fixture(autouse=True)
def _fresh_registry() -> None:
    """Reset the module-level registry before every test."""
    mcp_server._REGISTRY = JobRegistry()


def _fake_popen_blocking(delay_s: float, verdict: dict) -> MagicMock:
    """Build a Popen mock that blocks `delay_s` seconds before
    returning a successful claude-CLI JSON envelope. This lets us
    simulate slow workers without spawning real subprocesses.
    """
    fake = MagicMock(spec=subprocess.Popen)
    fake.poll.return_value = None  # still running until communicate finishes

    def _communicate(timeout: float | None = None) -> tuple[str, str]:
        time.sleep(delay_s)
        return (json.dumps({
            "is_error": False,
            "total_cost_usd": 0.01,
            "structured_output": verdict,
        }), "")

    fake.communicate.side_effect = _communicate
    fake.returncode = 0
    return fake


def _call_tool_sync(name: str, args: dict) -> dict:
    """Invoke an MCP tool handler from a synchronous test and return
    the parsed JSON payload.

    The mcp.Server stores the registered handlers in private attrs that
    differ slightly across the 1.x line; we look them up via the public
    `_call_tool` symbol that the module exposes for this purpose.
    """
    handler = mcp_server._call_tool_impl
    result = asyncio.run(handler(name, args))
    assert len(result) == 1
    return json.loads(result[0].text)


def test_start_returns_job_id_under_100ms(tmp_path: Path) -> None:
    """The whole point of the async pattern: start must NOT block on
    worker spawn. It returns in under 100 ms even when the worker
    will take 30+ seconds.
    """
    fake = _fake_popen_blocking(delay_s=30.0, verdict={
        "claim_holds": True, "confidence": 0.9, "evidence": "x",
        "counterexample_found": False,
    })
    with patch("critic_orchestrator.orchestrator.subprocess.Popen",
               return_value=fake):
        t0 = time.perf_counter()
        resp = _call_tool_sync("start_adversarial_review", {
            "claim": "c", "diff_summary": "d",
            "project_dir": str(tmp_path),
        })
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert "job_id" in resp
    assert resp["status"] == "running"
    assert resp["n_workers"] == 1  # only counterexample (no test_path / fixed_function)
    assert elapsed_ms < 100, (
        f"start_adversarial_review took {elapsed_ms:.0f}ms — must be <100ms"
    )


def test_poll_progresses_from_running_to_done(tmp_path: Path) -> None:
    """A short fake worker (50ms) lets us observe running → done."""
    fake = _fake_popen_blocking(delay_s=0.05, verdict={
        "claim_holds": True, "confidence": 0.9, "evidence": "x",
        "counterexample_found": False,
    })
    with patch("critic_orchestrator.orchestrator.subprocess.Popen",
               return_value=fake):
        started = _call_tool_sync("start_adversarial_review", {
            "claim": "c", "diff_summary": "d",
            "project_dir": str(tmp_path),
        })
        job_id = started["job_id"]
        # Poll until done — bounded budget to avoid a hang on regression.
        deadline = time.time() + 5.0
        final = None
        while time.time() < deadline:
            polled = _call_tool_sync("poll_adversarial_review",
                                     {"job_id": job_id})
            if polled["status"] != "running":
                final = polled
                break
            time.sleep(0.05)
        assert final is not None, "job never left running state in 5s"
        assert final["status"] == "done"
        assert final["result"]["consensus"] == "claim_holds"
        assert final["result"]["votes"]["hold"] == 1


def test_poll_unknown_job_id_returns_error(tmp_path: Path) -> None:
    resp = _call_tool_sync("poll_adversarial_review",
                            {"job_id": "does-not-exist"})
    assert "error" in resp
    assert "unknown" in resp["error"].lower() or "not found" in resp["error"].lower()


def test_cancel_kills_in_flight_workers(tmp_path: Path) -> None:
    """A long fake worker (5s) lets us cancel mid-flight and verify
    the Popen handle was killed and the job marked cancelled.
    """
    fake = MagicMock(spec=subprocess.Popen)
    fake.poll.return_value = None  # still running

    # Communicate blocks until kill() is called via a threading.Event.
    import threading
    killed_event = threading.Event()

    def _communicate(timeout: float | None = None) -> tuple[str, str]:
        if killed_event.wait(timeout=10.0):
            raise subprocess.TimeoutExpired(cmd="claude", timeout=timeout or 0)
        return ("", "")

    def _kill() -> None:
        killed_event.set()

    fake.communicate.side_effect = _communicate
    fake.kill.side_effect = _kill
    fake.returncode = -9

    with patch("critic_orchestrator.orchestrator.subprocess.Popen",
               return_value=fake):
        started = _call_tool_sync("start_adversarial_review", {
            "claim": "c", "diff_summary": "d",
            "project_dir": str(tmp_path),
        })
        job_id = started["job_id"]
        # Give the background task a moment to register the Popen handle.
        time.sleep(0.2)
        resp = _call_tool_sync("cancel_adversarial_review",
                                {"job_id": job_id})

    assert resp["status"] == "cancelled"
    assert resp["killed_workers"] >= 1
    fake.kill.assert_called()


def test_cancel_unknown_job_id_returns_error(tmp_path: Path) -> None:
    resp = _call_tool_sync("cancel_adversarial_review",
                            {"job_id": "does-not-exist"})
    assert "error" in resp


def test_list_adversarial_reviews_returns_known_jobs(tmp_path: Path) -> None:
    fake = _fake_popen_blocking(delay_s=0.05, verdict={
        "claim_holds": True, "confidence": 0.9, "evidence": "x",
        "counterexample_found": False,
    })
    with patch("critic_orchestrator.orchestrator.subprocess.Popen",
               return_value=fake):
        s1 = _call_tool_sync("start_adversarial_review", {
            "claim": "c1", "diff_summary": "d", "project_dir": str(tmp_path),
        })
        s2 = _call_tool_sync("start_adversarial_review", {
            "claim": "c2", "diff_summary": "d", "project_dir": str(tmp_path),
        })
        resp = _call_tool_sync("list_adversarial_reviews", {})
    ids = {j["job_id"] for j in resp["jobs"]}
    assert s1["job_id"] in ids
    assert s2["job_id"] in ids


def test_legacy_force_adversarial_review_still_works(tmp_path: Path) -> None:
    """Backwards compat: the original synchronous tool must keep its
    contract — block until done, return the full report inline.
    """
    fake = _fake_popen_blocking(delay_s=0.05, verdict={
        "claim_holds": True, "confidence": 0.9, "evidence": "x",
        "counterexample_found": False,
    })
    with patch("critic_orchestrator.orchestrator.subprocess.Popen",
               return_value=fake):
        resp = _call_tool_sync("force_adversarial_review", {
            "claim": "c", "diff_summary": "d",
            "project_dir": str(tmp_path),
        })
    assert resp["consensus"] == "claim_holds"
    assert resp["votes"]["hold"] == 1
    assert "wall_duration_ms" in resp


def test_start_validates_required_fields(tmp_path: Path) -> None:
    resp = _call_tool_sync("start_adversarial_review", {
        "claim": "", "diff_summary": "",
    })
    assert "error" in resp


def test_start_with_three_workers_when_test_and_fn_given(tmp_path: Path) -> None:
    """When test_path and fixed_function are passed, all three default
    workers should be spawned (falsification + caller + counterexample).
    """
    fake = _fake_popen_blocking(delay_s=0.05, verdict={
        "claim_holds": True, "confidence": 0.9, "evidence": "x",
        "test_falsifies_master": True,
        "counterexample_found": False,
        "production_caller_exists": True,
        "caller_paths": [],
    })
    with patch("critic_orchestrator.orchestrator.subprocess.Popen",
               return_value=fake):
        resp = _call_tool_sync("start_adversarial_review", {
            "claim": "c", "diff_summary": "d",
            "test_path": "tests/test_foo.py::test_bar",
            "fixed_function": "foo",
            "project_dir": str(tmp_path),
        })
    assert resp["n_workers"] == 3
