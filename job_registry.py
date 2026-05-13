"""In-process registry of long-running adversarial reviews.

The synchronous force_adversarial_review tool blocks for 30-180s on MCP
stdio while the worker subprocesses run. The Claude Code MCP client has
a ~60s deadline by default, so any review touching the falsification
worker (git stash + pytest + git stash pop + pytest) exceeds the budget
and the client kills the connection with -32001 — leaving the stdio of
that session in a stale state until restart.

This module provides a non-blocking alternative: a Job dataclass that
runs the review in a background asyncio task and stores results in a
registry. Three MCP tools expose the lifecycle:

    start_adversarial_review(...)    # returns job_id immediately
    poll_adversarial_review(job_id)  # returns status + result if done
    cancel_adversarial_review(job_id)

A TTL sweeper drops finished jobs after JOB_TTL_S (default 1h) to
prevent unbounded growth. The registry is in-process — a server
restart loses all jobs (acceptable for an adversarial-review tool;
the client can simply re-issue).
"""
from __future__ import annotations

import secrets
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any

from .orchestrator import CriticReport, WorkerSpec

from .orchestrator import kill_process_tree

JOB_TTL_S: float = 3600.0
"""How long terminal-state jobs are retained before being GC-ed."""


@dataclass
class Job:
    """One in-flight or completed adversarial review."""

    id: str
    claim: str
    project_dir: Path
    workers: list[WorkerSpec]
    timeout_s: int
    status: str  # "running" | "done" | "failed" | "cancelled"
    started_at: float
    ended_at: float | None = None
    report: CriticReport | None = None
    error: str | None = None
    popen_handles: list[subprocess.Popen[bytes]] = field(default_factory=list)
    # Cancellation flag. Set by JobRegistry.cancel BEFORE iterating
    # popen_handles. Worker threads check this between key operations
    # (pre-Popen, post-Popen, post-append) so a cancel issued during
    # the spawn window aborts the worker before the subprocess starts
    # doing real work (git stash + pytest + LLM call).
    aborted: bool = False

    @property
    def is_terminal(self) -> bool:
        return self.status in ("done", "failed", "cancelled")

    @property
    def elapsed_s(self) -> float:
        end = self.ended_at if self.is_terminal else time.time()
        return end - self.started_at

    def as_dict(self, include_report: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {
            "job_id": self.id,
            "status": self.status,
            "elapsed_s": round(self.elapsed_s, 3),
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "n_workers": len(self.workers),
            "claim": self.claim[:200],
        }
        if include_report and self.report is not None:
            out["result"] = self.report.as_dict()
        if self.error:
            out["error"] = self.error
        return out


class JobRegistry:
    """Thread-safe in-memory job store with TTL-based GC.

    The registry is intentionally simple: a dict keyed by job_id under a
    re-entrant lock. Every mutating call sweeps expired terminal jobs
    so memory cannot grow unbounded across a long-lived MCP server.
    """

    def __init__(self, ttl_s: float = JOB_TTL_S) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = RLock()
        self._ttl_s = ttl_s

    def create(
        self,
        *,
        claim: str,
        project_dir: Path,
        workers: list[WorkerSpec],
        timeout_s: int,
    ) -> Job:
        job_id = secrets.token_hex(8)
        with self._lock:
            self._gc_expired_locked()
            job = Job(
                id=job_id, claim=claim, project_dir=project_dir,
                workers=workers, timeout_s=timeout_s,
                status="running", started_at=time.time(),
            )
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            self._gc_expired_locked()
            return self._jobs.get(job_id)

    def mark_done(self, job: Job, report: CriticReport) -> None:
        with self._lock:
            if job.is_terminal:
                return
            job.report = report
            job.status = "done"
            job.ended_at = time.time()

    def mark_failed(self, job: Job, error: str) -> None:
        with self._lock:
            if job.is_terminal:
                return
            job.error = error
            job.status = "failed"
            job.ended_at = time.time()

    def mark_cancelled(self, job: Job) -> None:
        with self._lock:
            if job.is_terminal:
                return
            job.status = "cancelled"
            job.ended_at = time.time()

    def cancel(self, job_id: str) -> int:
        """Cancel a running job. Returns number of worker subprocesses
        actually killed (excludes those that had already exited).

        Sets `job.aborted=True` FIRST so any worker thread currently
        spawning a new Popen sees the flag and aborts before the real
        work starts. Then iterates the already-registered handles and
        kills each (entire process tree, see `_kill_process_tree`).
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.is_terminal:
                return 0
            # Set BEFORE iterating: a worker thread doing
            # popen_sink.append(...) checks `aborted` right after and
            # kills its own Popen if set. Without this flag, the
            # cancel races against the spawn and can leak a subprocess
            # that runs for the full timeout (180s).
            job.aborted = True
            killed = 0
            for handle in list(job.popen_handles):
                try:
                    if handle.poll() is None:
                        kill_process_tree(handle)
                        killed += 1
                except Exception:
                    # A handle in an indeterminate state (closed pipe,
                    # already-reaped PID) is treated as "no longer
                    # running" — don't abort the cancel sweep.
                    pass
            self.mark_cancelled(job)
            return killed

    def list_active(self) -> list[Job]:
        with self._lock:
            self._gc_expired_locked()
            return [j for j in self._jobs.values() if not j.is_terminal]

    def list_all(self) -> list[Job]:
        with self._lock:
            self._gc_expired_locked()
            return list(self._jobs.values())

    def _gc_expired_locked(self) -> None:
        now = time.time()
        expired = [
            jid for jid, j in self._jobs.items()
            if j.is_terminal
            and j.ended_at is not None
            and (now - j.ended_at) > self._ttl_s
        ]
        for jid in expired:
            self._jobs.pop(jid, None)


__all__ = ["JOB_TTL_S", "Job", "JobRegistry"]
