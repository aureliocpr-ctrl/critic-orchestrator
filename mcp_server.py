"""MCP server exposing the critic-orchestrator.

Five tools:

  * `force_adversarial_review` — legacy SYNCHRONOUS path. Blocks until
    every worker finishes. Kept for backwards compatibility, but the
    MCP client (Claude Code) has a ~60s deadline by default, so any
    review that touches the falsification worker (git stash + pytest)
    will exceed the budget and get killed with `-32001 MCP timeout`.

  * `start_adversarial_review`, `poll_adversarial_review`,
    `cancel_adversarial_review`, `list_adversarial_reviews` — the new
    async-job pattern. `start` returns a `job_id` in <100 ms; the
    caller polls until `status="done"` and then reads the report. No
    MCP-level timeout pressure.

Run as: `python -m critic_orchestrator.mcp_server`

Register in ~/.mcp.json:
    {
      "mcpServers": {
        "critic-orchestrator": {
          "command": "python",
          "args": ["-m", "critic_orchestrator.mcp_server"],
          "env": {"PYTHONPATH": "C:/Users/aurel/.claude"}
        }
      }
    }
"""
from __future__ import annotations

import asyncio
import atexit
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import mcp.types as t
from mcp.server import Server
from mcp.server.stdio import stdio_server

from .backends import make_backend_from_env
from .default_workers import build_default_workers
from .job_registry import Job, JobRegistry
from .orchestrator import adversarial_review


server: Server = Server("critic-orchestrator")

# Module-level registry shared by all async-job tools. Tests reset this
# by reassigning the attribute (see tests/test_mcp_async_tools.py).
_REGISTRY: JobRegistry = JobRegistry()

# Dedicated executor for background review jobs. We deliberately do NOT
# use asyncio's default executor: `asyncio.run()` calls
# `loop.shutdown_default_executor()` on exit and waits for it, which
# would block any caller — including unit tests — until the in-flight
# review finishes. A module-level ThreadPoolExecutor survives across
# `asyncio.run` invocations and lets background work outlive the
# request that started it.
#
# Max workers cap: 8 in-flight reviews × 3 worker subprocesses each =
# 24 sub-claude processes, which fits comfortably on a developer
# laptop. Reviews queue beyond that.
_EXECUTOR: ThreadPoolExecutor = ThreadPoolExecutor(
    max_workers=8, thread_name_prefix="critic-job-",
)

# Best-effort shutdown so a SIGINT-killed MCP server does not strand
# in-flight worker subprocesses. `cancel_futures=True` drops pending
# jobs from the queue; running threads are not preempted — they will
# observe the parent exit when their next `communicate()` returns.
atexit.register(lambda: _EXECUTOR.shutdown(wait=False, cancel_futures=True))


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

_SHARED_INPUT_PROPS: dict[str, Any] = {
    "claim": {
        "type": "string",
        "description": "1-2 sentence statement of what the main agent believes was fixed or verified.",
    },
    "diff_summary": {
        "type": "string",
        "description": "Human-readable summary of what changed (filenames, function names, line ranges).",
    },
    "test_path": {
        "type": "string",
        "description": "Pytest path or file::function selector that should fail on master pre-fix. Omit to skip the falsification worker.",
    },
    "fixed_function": {
        "type": "string",
        "description": "Name of the function/method the fix targets. Omit to skip the caller-verification worker.",
    },
    "project_dir": {
        "type": "string",
        "description": "Absolute path to the project root. Workers run with this as cwd. Defaults to $PWD.",
    },
    "timeout_s": {
        "type": "integer",
        "minimum": 30, "maximum": 600, "default": 180,
        "description": "Per-worker subprocess timeout (server-side). Independent of any MCP client timeout.",
    },
}

_REQUIRED_INPUT = ["claim", "diff_summary"]


def _force_review_tool() -> t.Tool:
    return t.Tool(
        name="force_adversarial_review",
        description=(
            "[LEGACY SYNCHRONOUS] Spawn N fresh Claude CLI workers in "
            "parallel to adversarially verify a claim/fix and BLOCK "
            "until they finish. Kept for backwards compatibility, but "
            "the MCP client times out after ~60s by default — use "
            "`start_adversarial_review` + `poll_adversarial_review` "
            "for any review involving the falsification worker."
        ),
        inputSchema={
            "type": "object",
            "properties": _SHARED_INPUT_PROPS,
            "required": _REQUIRED_INPUT,
        },
    )


def _start_review_tool() -> t.Tool:
    return t.Tool(
        name="start_adversarial_review",
        description=(
            "Start an adversarial review in the background and return "
            "a `job_id` in <100 ms. Use `poll_adversarial_review` to "
            "check progress and retrieve the result, and "
            "`cancel_adversarial_review` to abort. Bypasses the MCP "
            "client timeout because the call returns immediately. "
            "Same worker semantics as `force_adversarial_review`."
        ),
        inputSchema={
            "type": "object",
            "properties": _SHARED_INPUT_PROPS,
            "required": _REQUIRED_INPUT,
        },
    )


def _poll_review_tool() -> t.Tool:
    return t.Tool(
        name="poll_adversarial_review",
        description=(
            "Check the status of a background adversarial review. "
            "Returns `status` ∈ {running, done, failed, cancelled}, "
            "`elapsed_s`, and — when status is `done` — the full "
            "report under `result`."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "job_id": {"type": "string"},
            },
            "required": ["job_id"],
        },
    )


def _cancel_review_tool() -> t.Tool:
    return t.Tool(
        name="cancel_adversarial_review",
        description=(
            "Cancel a running adversarial review. Kills every "
            "in-flight worker subprocess and marks the job "
            "`cancelled`. Idempotent on already-terminal jobs."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "job_id": {"type": "string"},
            },
            "required": ["job_id"],
        },
    )


def _list_reviews_tool() -> t.Tool:
    return t.Tool(
        name="list_adversarial_reviews",
        description=(
            "List all adversarial-review jobs known to this server "
            "(running + recently completed). Useful for inspection "
            "and debugging stuck jobs."
        ),
        inputSchema={"type": "object", "properties": {}},
    )


@server.list_tools()
async def _list_tools() -> list[t.Tool]:
    return [
        _force_review_tool(),
        _start_review_tool(),
        _poll_review_tool(),
        _cancel_review_tool(),
        _list_reviews_tool(),
    ]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _parse_common(arguments: dict[str, Any]) -> dict[str, Any] | str:
    """Validate the inputs shared across force/start. Returns either a
    dict of parsed values or an error string.
    """
    claim = str(arguments.get("claim", "")).strip()
    diff_summary = str(arguments.get("diff_summary", "")).strip()
    if not claim or not diff_summary:
        return "claim and diff_summary are required"
    test_path = arguments.get("test_path") or None
    fixed_function = arguments.get("fixed_function") or None
    project_dir = Path(arguments.get("project_dir") or os.getcwd()).resolve()
    timeout_s = int(arguments.get("timeout_s") or 180)
    timeout_s = max(30, min(600, timeout_s))
    workers = build_default_workers(
        claim=claim, diff_summary=diff_summary,
        test_path=test_path, fixed_function=fixed_function,
    )
    return {
        "claim": claim,
        "diff_summary": diff_summary,
        "test_path": test_path,
        "fixed_function": fixed_function,
        "project_dir": project_dir,
        "timeout_s": timeout_s,
        "workers": workers,
    }


def _run_review_in_thread(job: Job, backend: Any | None = None) -> None:
    """Body of the background task: run adversarial_review and store
    the result in the job. Called inside an executor thread.

    `backend` is the provider backend selected at start time (None = the
    built-in Claude CLI path). We use `BaseException` rather than
    `Exception` so a worker thread interrupted by KeyboardInterrupt /
    SystemExit / CancelledError leaves the job in a `failed` terminal
    state instead of stuck on `running` forever (where `poll` would never
    resolve). The exception is re-raised so the thread itself surfaces the
    signal to whatever upstream watcher is listening.
    """
    try:
        report = adversarial_review(
            claim=job.claim,
            project_dir=job.project_dir,
            workers=job.workers,
            timeout=job.timeout_s,
            popen_sink=job.popen_handles,
            cancel_check=lambda: job.aborted,
            backend=backend,
        )
    except Exception as exc:
        _REGISTRY.mark_failed(job, f"orchestrator error: {exc!r}")
        return
    except BaseException as exc:  # pragma: no cover - signal path
        _REGISTRY.mark_failed(job, f"interrupted: {type(exc).__name__}")
        raise
    # If the job was cancelled mid-flight, leave its terminal state alone.
    if not job.is_terminal:
        _REGISTRY.mark_done(job, report)


async def _call_tool_impl(
    name: str, arguments: dict[str, Any],
) -> list[t.TextContent]:
    """Dispatch a single MCP tool call. Exposed (with leading underscore
    on the module) for direct invocation from unit tests.
    """
    if name == "force_adversarial_review":
        parsed = _parse_common(arguments)
        if isinstance(parsed, str):
            return [t.TextContent(type="text",
                                   text=json.dumps({"error": parsed}))]
        try:
            backend = make_backend_from_env()
        except ValueError as exc:
            return [t.TextContent(type="text",
                                   text=json.dumps({"error": str(exc)}))]
        loop = asyncio.get_running_loop()
        report = await loop.run_in_executor(
            _EXECUTOR,
            lambda: adversarial_review(
                claim=parsed["claim"],
                project_dir=parsed["project_dir"],
                workers=parsed["workers"],
                timeout=parsed["timeout_s"],
                backend=backend,
            ),
        )
        return [t.TextContent(type="text",
                               text=json.dumps(report.as_dict(), indent=2))]

    if name == "start_adversarial_review":
        parsed = _parse_common(arguments)
        if isinstance(parsed, str):
            return [t.TextContent(type="text",
                                   text=json.dumps({"error": parsed}))]
        try:
            backend = make_backend_from_env()
        except ValueError as exc:
            return [t.TextContent(type="text",
                                   text=json.dumps({"error": str(exc)}))]
        job = _REGISTRY.create(
            claim=parsed["claim"],
            project_dir=parsed["project_dir"],
            workers=parsed["workers"],
            timeout_s=parsed["timeout_s"],
        )
        # Fire-and-forget on the module-level executor. We deliberately
        # bypass `loop.run_in_executor(None, ...)` so the work survives
        # the `asyncio.run()` shutdown sweep that would otherwise block
        # this handler until the review finishes. The job's terminal
        # state is set by _run_review_in_thread itself; poll reads it.
        _EXECUTOR.submit(_run_review_in_thread, job, backend)
        return [t.TextContent(type="text",
                               text=json.dumps(job.as_dict()))]

    if name == "poll_adversarial_review":
        job_id = str(arguments.get("job_id", "")).strip()
        if not job_id:
            return [t.TextContent(type="text",
                                   text=json.dumps({"error": "job_id required"}))]
        job = _REGISTRY.get(job_id)
        if job is None:
            return [t.TextContent(type="text", text=json.dumps({
                "error": f"unknown job_id: {job_id}",
            }))]
        return [t.TextContent(type="text",
                               text=json.dumps(job.as_dict(include_report=True),
                                               indent=2))]

    if name == "cancel_adversarial_review":
        job_id = str(arguments.get("job_id", "")).strip()
        if not job_id:
            return [t.TextContent(type="text",
                                   text=json.dumps({"error": "job_id required"}))]
        job = _REGISTRY.get(job_id)
        if job is None:
            return [t.TextContent(type="text", text=json.dumps({
                "error": f"unknown job_id: {job_id}",
            }))]
        killed = _REGISTRY.cancel(job_id)
        # Re-fetch to surface the new status.
        job = _REGISTRY.get(job_id)
        out = job.as_dict() if job else {"job_id": job_id, "status": "cancelled"}
        out["killed_workers"] = killed
        return [t.TextContent(type="text", text=json.dumps(out))]

    if name == "list_adversarial_reviews":
        jobs = [j.as_dict() for j in _REGISTRY.list_all()]
        # Newest first by start time.
        jobs.sort(key=lambda d: d.get("started_at", 0.0), reverse=True)
        return [t.TextContent(type="text",
                               text=json.dumps({"jobs": jobs}, indent=2))]

    return [t.TextContent(type="text",
                           text=json.dumps({"error": f"unknown tool: {name}"}))]


@server.call_tool()
async def _call_tool(name: str, arguments: dict[str, Any]) -> list[t.TextContent]:
    return await _call_tool_impl(name, arguments)


async def _serve() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream,
                            server.create_initialization_options())


def main() -> None:
    asyncio.run(_serve())


if __name__ == "__main__":
    sys.exit(main() or 0)
