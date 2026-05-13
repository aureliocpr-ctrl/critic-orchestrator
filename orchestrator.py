"""Core orchestrator: spawn N `claude --print` workers in parallel,
collect their JSON-schema-constrained verdicts, aggregate.

The orchestrator is domain-agnostic. The caller passes:
  - the claim being verified (string, just for record-keeping)
  - a list of WorkerSpec — each one names a worker role, gives it a
    prompt template, and declares the JSON schema its output must
    obey
  - project_dir (the cwd the workers see)
  - timeout, optional MCP config passed to workers

Each worker runs an isolated subprocess of the `claude` CLI binary.
The subprocess inherits the user's Claude Code authentication
(subscription, not API key). No Anthropic Python SDK is imported.

Subprocesses are invoked via subprocess.run with argv list (no shell
interpolation, no command-injection surface). Concurrency is achieved
with ThreadPoolExecutor since each worker is I/O bound on a separate
OS process.
"""
from __future__ import annotations

import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

try:
    import psutil  # type: ignore[import-not-found]
    _HAS_PSUTIL = True
except ImportError:  # pragma: no cover - psutil ships with the package
    _HAS_PSUTIL = False


def kill_process_tree(proc: subprocess.Popen) -> None:
    """Terminate a subprocess and every descendant it spawned.

    `Popen.kill()` only signals the direct child. The `claude` CLI is
    a thin Node.js launcher; killing the launcher leaves the node
    runtime running until its socket-read times out, keeping a worker
    (git stash + pytest) alive minutes after we asked it to stop. This
    foot-gun was surfaced by the critic-orchestrator's own
    counterexample worker during adversarial review of the async-job
    fix (confidence 0.92, "missing process-group / Job Object
    semantics").

    Best-effort: if psutil cannot enumerate children (process already
    gone, permission denied, etc.), we fall back to `proc.kill()` so
    we at least signal the parent.
    """
    if _HAS_PSUTIL:
        try:
            parent = psutil.Process(proc.pid)
            for child in parent.children(recursive=True):
                try:
                    child.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            try:
                parent.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            return
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        except Exception:
            pass
    try:
        proc.kill()
    except Exception:
        pass


@dataclass(frozen=True)
class WorkerSpec:
    """Recipe for a single adversarial worker."""

    name: str
    prompt: str
    schema: dict[str, Any]
    # Extra CLI flags the caller wants passed verbatim. Useful for
    # things like `--add-dir /other/path` or `--allowedTools Read`.
    extra_args: tuple[str, ...] = field(default_factory=tuple)
    # Permission mode for the worker. "plan" is read-only (safe
    # default); "acceptEdits" allows the worker to run side-effecting
    # commands like `git stash` and `pytest` — required by the
    # falsification worker, not by the others.
    permission_mode: str = "plan"


@dataclass
class WorkerVerdict:
    """One worker's outcome."""

    name: str
    verdict: dict[str, Any] | None
    error: str | None
    cost_usd: float
    duration_ms: int
    raw_stdout_preview: str = ""

    @property
    def ok(self) -> bool:
        return self.verdict is not None and self.error is None


@dataclass
class CriticReport:
    """Aggregated review outcome."""

    claim: str
    workers: list[WorkerVerdict]
    consensus: str  # "claim_holds" | "claim_fails" | "split" | "undecided"
    votes_hold: int
    votes_fail: int
    votes_invalid: int
    total_cost_usd: float
    wall_duration_ms: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "claim": self.claim,
            "consensus": self.consensus,
            "votes": {
                "hold": self.votes_hold,
                "fail": self.votes_fail,
                "invalid": self.votes_invalid,
            },
            "total_cost_usd": self.total_cost_usd,
            "wall_duration_ms": self.wall_duration_ms,
            "workers": [
                {
                    "name": w.name,
                    "ok": w.ok,
                    "verdict": w.verdict,
                    "error": w.error,
                    "cost_usd": w.cost_usd,
                    "duration_ms": w.duration_ms,
                }
                for w in self.workers
            ],
        }


# Keys we recognise as "the boolean that says whether the claim holds".
# The first key found in the verdict dict wins. False means the claim
# fails; true means it holds.
_CLAIM_HOLDS_KEYS: tuple[str, ...] = (
    "claim_holds",
    "claim_valid",
    "fix_is_real",
    "verdict",
)
# Keys that mean the OPPOSITE — true means the claim fails.
_CLAIM_FAILS_KEYS: tuple[str, ...] = (
    "claim_is_theatre",
    "is_confabulation",
    "would_not_fail_pre_fix",
)


def _extract_vote(verdict: dict[str, Any]) -> bool | None:
    """Map a worker's structured output to a bool vote.

    Returns True if the worker says the claim holds, False if it
    says the claim fails, None if no recognised vote key is present.
    """
    for key in _CLAIM_HOLDS_KEYS:
        if key in verdict and isinstance(verdict[key], bool):
            return verdict[key]
    for key in _CLAIM_FAILS_KEYS:
        if key in verdict and isinstance(verdict[key], bool):
            return not verdict[key]
    return None


def _spawn_worker(
    spec: WorkerSpec,
    project_dir: Path,
    timeout: int,
    extra_mcp: dict[str, Any] | None,
    *,
    popen_sink: list[subprocess.Popen] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> WorkerVerdict:
    """Spawn one `claude --print` subprocess (no shell, argv list),
    return its WorkerVerdict. Synchronous — meant to be run inside a
    ThreadPoolExecutor for concurrency across workers.

    If `popen_sink` is provided, the Popen handle is appended to it
    BEFORE `communicate()` blocks. This is the cancellability contract
    used by JobRegistry.cancel — an external thread can iterate the
    sink and call .kill() on each handle to abort an in-flight review.

    If `cancel_check` is provided, the function consults it (a 0-arg
    callable returning True iff the job has been cancelled) at three
    points: before Popen, immediately after Popen, and after appending
    to the sink. This closes the race window in which a cancel issued
    during spawn would otherwise leak a subprocess that runs until
    its natural timeout.
    """
    mcp_config = {"mcpServers": extra_mcp or {}}
    cmd: list[str] = [
        "claude",
        "--print",
        "--output-format", "json",
        "--strict-mcp-config",
        "--mcp-config", json.dumps(mcp_config),
        # Bypass permission checks. In non-interactive --print mode
        # there's no TTY to confirm tool-use prompts, so a worker
        # that wants to call Bash/Read/Grep hangs indefinitely
        # waiting for a confirmation that never comes. Tools are
        # already gated by the per-worker --allowedTools whitelist,
        # so this only skips the "may I?" dialog, not the tool set.
        "--dangerously-skip-permissions",
        "--json-schema", json.dumps(spec.schema),
        "--no-session-persistence",
        *spec.extra_args,
        # `--` terminates flag parsing. Without it, variadic flags
        # like `--allowedTools <tools...>` swallow the prompt as
        # an extra tool name and `claude --print` errors out with
        # "Input must be provided either through stdin or as a
        # prompt argument when using --print".
        "--",
        spec.prompt,
    ]
    t0 = time.perf_counter()
    # Debug: dump the worker command + env to a log so we can inspect
    # what the subprocess saw when it's hung. Written before the spawn
    # so a hung worker still leaves a trace.
    import os as _os
    log_dir = Path(_os.environ.get("TEMP", "/tmp")) / "critic_orchestrator"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"worker_{spec.name}_{int(t0*1000)}.log"
    try:
        with open(log_path, "w", encoding="utf-8") as logf:
            logf.write(f"CWD: {project_dir}\n")
            logf.write(f"CMD: {' '.join(repr(c) for c in cmd)}\n\n")
    except Exception:
        pass

    def _finalize(end_state: str, error: str | None,
                   cost: float, wall_ms: int) -> None:
        """Append the completion event to the worker log."""
        try:
            with open(log_path, "a", encoding="utf-8") as logf:
                logf.write(
                    f"END: state={end_state} duration_ms={wall_ms} "
                    f"cost_usd={cost:.4f}"
                )
                if error:
                    logf.write(f" error={error[:200]!r}")
                logf.write("\n")
        except Exception:
            pass

    # PRE-SPAWN cancel check. If the job was already aborted (cancel
    # arrived before the worker thread reached this line) we avoid
    # spawning a real subprocess and the associated LLM cost entirely.
    if cancel_check is not None and cancel_check():
        wall_ms = int((time.perf_counter() - t0) * 1000)
        _finalize("cancelled_pre_spawn", None, 0.0, wall_ms)
        return WorkerVerdict(
            name=spec.name, verdict=None,
            error="cancelled before spawn", cost_usd=0.0,
            duration_ms=wall_ms,
        )

    try:
        proc = subprocess.Popen(  # noqa: S603 (argv list, no shell)
            cmd,
            cwd=str(project_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            # CRITICAL: detach stdin. Without this, the worker
            # subprocess inherits our stdin — which, when the
            # orchestrator runs inside the critic-orchestrator MCP
            # server, is the JSON-RPC pipe to the parent claude
            # session. `claude --print` falls back to reading the
            # prompt from stdin when the positional prompt is
            # missing (e.g. swallowed by a variadic flag), and the
            # MCP pipe never delivers a usable string → the worker
            # waits indefinitely until the timeout hits. DEVNULL
            # closes that path so the worker either uses the
            # positional prompt or errors out immediately.
            stdin=subprocess.DEVNULL,
            shell=False,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError:
        wall_ms = int((time.perf_counter() - t0) * 1000)
        _finalize("error", "claude binary not on PATH", 0.0, wall_ms)
        return WorkerVerdict(
            name=spec.name, verdict=None,
            error="`claude` binary not on PATH",
            cost_usd=0.0, duration_ms=wall_ms,
        )

    # Register the handle BEFORE awaiting communicate() so an external
    # cancel can reach it even while the thread is parked.
    if popen_sink is not None:
        popen_sink.append(proc)

    # POST-REGISTER cancel check. Closes the race where cancel runs
    # between Popen() and the append above: at that point the cancel
    # loop saw an empty sink, marked the job cancelled, and walked
    # away — without this check the worker would happily keep going.
    # Now we tree-kill the just-spawned subprocess and bail out.
    if cancel_check is not None and cancel_check():
        kill_process_tree(proc)
        try:
            proc.communicate(timeout=5)
        except Exception:
            pass
        wall_ms = int((time.perf_counter() - t0) * 1000)
        _finalize("cancelled_post_spawn", None, 0.0, wall_ms)
        return WorkerVerdict(
            name=spec.name, verdict=None,
            error="cancelled after spawn", cost_usd=0.0,
            duration_ms=wall_ms,
        )

    try:
        stdout_raw, stderr_raw = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        # Kill the subprocess AND its descendants (claude → node) so
        # no orphaned grand-child keeps grinding pytest in the
        # background. communicate() with no timeout after kill drains
        # the pipes; we discard the output since the worker exceeded
        # its budget.
        kill_process_tree(proc)
        try:
            proc.communicate(timeout=5)
        except Exception:
            pass
        wall_ms = int((time.perf_counter() - t0) * 1000)
        _finalize("timeout", f"after {timeout}s", 0.0, wall_ms)
        return WorkerVerdict(
            name=spec.name, verdict=None,
            error=f"timeout after {timeout}s",
            cost_usd=0.0, duration_ms=wall_ms,
        )

    wall_ms = int((time.perf_counter() - t0) * 1000)
    stdout = (stdout_raw or "").strip()
    stderr = (stderr_raw or "").strip()
    if not stdout:
        err = f"empty stdout (rc={proc.returncode}, stderr={stderr[:200]})"
        _finalize("empty_stdout", err, 0.0, wall_ms)
        return WorkerVerdict(
            name=spec.name, verdict=None,
            error=err, cost_usd=0.0, duration_ms=wall_ms,
        )
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as parse_err:
        err = f"json_decode: {parse_err}"
        _finalize("json_decode", err, 0.0, wall_ms)
        return WorkerVerdict(
            name=spec.name, verdict=None,
            error=err, cost_usd=0.0, duration_ms=wall_ms,
            raw_stdout_preview=stdout[:500],
        )
    cost = float(payload.get("total_cost_usd", 0.0))
    if payload.get("is_error"):
        err = str(payload.get("result") or payload.get("subtype"))[:200]
        _finalize("cli_is_error", err, cost, wall_ms)
        return WorkerVerdict(
            name=spec.name, verdict=None,
            error=err, cost_usd=cost, duration_ms=wall_ms,
        )
    verdict = payload.get("structured_output")
    if not isinstance(verdict, dict):
        err = "missing structured_output"
        _finalize("missing_structured", err, cost, wall_ms)
        return WorkerVerdict(
            name=spec.name, verdict=None,
            error=err, cost_usd=cost, duration_ms=wall_ms,
            raw_stdout_preview=stdout[:500],
        )
    _finalize("ok", None, cost, wall_ms)
    return WorkerVerdict(
        name=spec.name, verdict=verdict, error=None,
        cost_usd=cost, duration_ms=wall_ms,
    )


def adversarial_review(
    claim: str,
    project_dir: Path,
    workers: list[WorkerSpec],
    *,
    timeout: int = 180,
    extra_mcp: dict[str, Any] | None = None,
    max_parallel: int = 4,
    popen_sink: list[subprocess.Popen] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> CriticReport:
    """Run every worker in parallel and aggregate their votes.

    Workers are spawned through a ThreadPoolExecutor; the total wall
    time is roughly max(worker_duration), not the sum, as long as
    max_parallel >= len(workers).

    If `popen_sink` is provided, each worker's Popen handle is appended
    to it so JobRegistry.cancel can terminate the in-flight review.

    If `cancel_check` is provided, every worker consults it before and
    immediately after spawning its Popen — a cancel that arrives during
    the spawn window aborts the worker without leaking a subprocess.
    """
    if not workers:
        return CriticReport(
            claim=claim, workers=[], consensus="undecided",
            votes_hold=0, votes_fail=0, votes_invalid=0,
            total_cost_usd=0.0, wall_duration_ms=0,
        )
    t0 = time.perf_counter()
    n_parallel = max(1, min(max_parallel, len(workers)))
    with ThreadPoolExecutor(max_workers=n_parallel) as pool:
        futures = [
            pool.submit(_spawn_worker, spec, project_dir,
                          timeout, extra_mcp,
                          popen_sink=popen_sink,
                          cancel_check=cancel_check)
            for spec in workers
        ]
        results = [f.result() for f in futures]
    wall_ms = int((time.perf_counter() - t0) * 1000)

    votes_hold = 0
    votes_fail = 0
    votes_invalid = 0
    for w in results:
        if not w.ok:
            votes_invalid += 1
            continue
        vote = _extract_vote(w.verdict or {})
        if vote is True:
            votes_hold += 1
        elif vote is False:
            votes_fail += 1
        else:
            votes_invalid += 1

    if votes_hold == 0 and votes_fail == 0:
        consensus = "undecided"
    elif votes_fail > votes_hold:
        consensus = "claim_fails"
    elif votes_hold > votes_fail:
        consensus = "claim_holds"
    else:
        consensus = "split"

    return CriticReport(
        claim=claim, workers=list(results),
        consensus=consensus,
        votes_hold=votes_hold,
        votes_fail=votes_fail,
        votes_invalid=votes_invalid,
        total_cost_usd=sum(w.cost_usd for w in results),
        wall_duration_ms=wall_ms,
    )


__all__ = [
    "WorkerSpec", "WorkerVerdict", "CriticReport",
    "adversarial_review",
]
