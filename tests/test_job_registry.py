"""Unit tests for the JobRegistry async-job state machine.

These tests run against the registry in isolation (no real subprocess
spawn, no claude CLI calls). They pin the lifecycle invariants:

    created -> running -> {done | failed | cancelled}

and the cancellation contract (kill subprocess handles, transition once).
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from critic_orchestrator.job_registry import JOB_TTL_S, Job, JobRegistry
from critic_orchestrator.orchestrator import CriticReport, WorkerSpec


def _spec(name: str = "w") -> WorkerSpec:
    return WorkerSpec(name=name, prompt="p", schema={"type": "object"})


def _report(claim: str = "c") -> CriticReport:
    return CriticReport(
        claim=claim, workers=[], consensus="claim_holds",
        votes_hold=1, votes_fail=0, votes_invalid=0,
        total_cost_usd=0.0, wall_duration_ms=10,
    )


def test_create_returns_unique_running_job(tmp_path: Path) -> None:
    reg = JobRegistry()
    j1 = reg.create(claim="c1", project_dir=tmp_path,
                    workers=[_spec("w1")], timeout_s=60)
    j2 = reg.create(claim="c2", project_dir=tmp_path,
                    workers=[_spec("w2")], timeout_s=60)
    assert j1.id != j2.id
    assert j1.status == "running"
    assert j2.status == "running"
    assert reg.get(j1.id) is j1
    assert reg.get(j2.id) is j2


def test_get_unknown_id_returns_none(tmp_path: Path) -> None:
    reg = JobRegistry()
    assert reg.get("does-not-exist") is None


def test_mark_done_transitions_to_terminal(tmp_path: Path) -> None:
    reg = JobRegistry()
    j = reg.create(claim="c", project_dir=tmp_path,
                   workers=[_spec()], timeout_s=60)
    assert not j.is_terminal
    reg.mark_done(j, _report())
    assert j.status == "done"
    assert j.is_terminal
    assert j.ended_at is not None
    assert j.report is not None


def test_mark_failed_records_error(tmp_path: Path) -> None:
    reg = JobRegistry()
    j = reg.create(claim="c", project_dir=tmp_path,
                   workers=[_spec()], timeout_s=60)
    reg.mark_failed(j, "boom")
    assert j.status == "failed"
    assert j.error == "boom"
    assert j.is_terminal


def test_mark_cancelled_idempotent_on_terminal(tmp_path: Path) -> None:
    """Cancelling a finished job must not overwrite its terminal state.

    Rationale: a poll → done → cancel race must not zombify a completed
    review.
    """
    reg = JobRegistry()
    j = reg.create(claim="c", project_dir=tmp_path,
                   workers=[_spec()], timeout_s=60)
    reg.mark_done(j, _report())
    ended_before = j.ended_at
    reg.mark_cancelled(j)
    assert j.status == "done"
    assert j.ended_at == ended_before


def test_elapsed_s_grows_while_running(tmp_path: Path) -> None:
    reg = JobRegistry()
    j = reg.create(claim="c", project_dir=tmp_path,
                   workers=[_spec()], timeout_s=60)
    e1 = j.elapsed_s
    time.sleep(0.05)
    e2 = j.elapsed_s
    assert e2 > e1


def test_elapsed_s_freezes_after_terminal(tmp_path: Path) -> None:
    reg = JobRegistry()
    j = reg.create(claim="c", project_dir=tmp_path,
                   workers=[_spec()], timeout_s=60)
    reg.mark_done(j, _report())
    e1 = j.elapsed_s
    time.sleep(0.05)
    e2 = j.elapsed_s
    assert e1 == e2


def test_as_dict_omits_result_unless_requested(tmp_path: Path) -> None:
    reg = JobRegistry()
    j = reg.create(claim="c", project_dir=tmp_path,
                   workers=[_spec()], timeout_s=60)
    reg.mark_done(j, _report())
    d_short = j.as_dict(include_report=False)
    d_full = j.as_dict(include_report=True)
    assert "result" not in d_short
    assert "result" in d_full
    assert d_full["result"]["consensus"] == "claim_holds"


def test_gc_expired_drops_old_terminal_jobs(tmp_path: Path) -> None:
    reg = JobRegistry(ttl_s=0.0)
    j_old = reg.create(claim="c", project_dir=tmp_path,
                       workers=[_spec()], timeout_s=60)
    reg.mark_done(j_old, _report())
    # Force ended_at into the past so the TTL fires deterministically.
    j_old.ended_at = time.time() - 10.0
    # Triggering a new create() runs the GC sweep.
    reg.create(claim="fresh", project_dir=tmp_path,
               workers=[_spec()], timeout_s=60)
    assert reg.get(j_old.id) is None


def test_gc_preserves_running_jobs(tmp_path: Path) -> None:
    """A long-running job (no ended_at) must never be GC-ed."""
    reg = JobRegistry(ttl_s=0.0)
    j_running = reg.create(claim="c", project_dir=tmp_path,
                            workers=[_spec()], timeout_s=60)
    reg.create(claim="fresh", project_dir=tmp_path,
               workers=[_spec()], timeout_s=60)
    assert reg.get(j_running.id) is j_running


def test_list_active_excludes_terminal(tmp_path: Path) -> None:
    reg = JobRegistry()
    j_run = reg.create(claim="c", project_dir=tmp_path,
                       workers=[_spec()], timeout_s=60)
    j_done = reg.create(claim="c", project_dir=tmp_path,
                        workers=[_spec()], timeout_s=60)
    reg.mark_done(j_done, _report())
    active = reg.list_active()
    ids = {j.id for j in active}
    assert j_run.id in ids
    assert j_done.id not in ids


def test_cancel_kills_registered_popen_handles(tmp_path: Path) -> None:
    """When the registry cancels a job, every registered Popen handle
    must be terminated. The handle is recorded by `_spawn_worker` so
    the cancel path can stop the workers mid-flight.
    """
    reg = JobRegistry()
    j = reg.create(claim="c", project_dir=tmp_path,
                   workers=[_spec("w1"), _spec("w2")], timeout_s=60)
    fake1 = MagicMock(spec=subprocess.Popen)
    fake1.poll.return_value = None  # still running
    fake2 = MagicMock(spec=subprocess.Popen)
    fake2.poll.return_value = None
    j.popen_handles.extend([fake1, fake2])

    killed = reg.cancel(j.id)
    assert killed == 2
    fake1.kill.assert_called_once()
    fake2.kill.assert_called_once()
    assert j.status == "cancelled"


def test_cancel_skips_already_finished_handles(tmp_path: Path) -> None:
    """A handle whose poll() != None already exited — do not call kill."""
    reg = JobRegistry()
    j = reg.create(claim="c", project_dir=tmp_path,
                   workers=[_spec("w1")], timeout_s=60)
    fake_done = MagicMock(spec=subprocess.Popen)
    fake_done.poll.return_value = 0  # completed
    j.popen_handles.append(fake_done)

    killed = reg.cancel(j.id)
    assert killed == 0
    fake_done.kill.assert_not_called()
    assert j.status == "cancelled"


def test_cancel_unknown_id_returns_zero(tmp_path: Path) -> None:
    reg = JobRegistry()
    killed = reg.cancel("does-not-exist")
    assert killed == 0


def test_cancel_done_job_does_not_restate_it(tmp_path: Path) -> None:
    reg = JobRegistry()
    j = reg.create(claim="c", project_dir=tmp_path,
                   workers=[_spec()], timeout_s=60)
    reg.mark_done(j, _report())
    killed = reg.cancel(j.id)
    assert killed == 0
    assert j.status == "done"


def test_job_default_ttl_constant_is_one_hour() -> None:
    assert JOB_TTL_S == 3600.0
