# critic-orchestrator

> MCP server that spawns N fresh Claude CLI workers in parallel to **adversarially verify** a claim/fix. Layer 3 of an anti-confabulation system for Claude Code.

**Status**: production, post-hardening 2026-05-13. 53/53 tests green.

---

## What it does

You just told Claude "I fixed the bug — the test now passes." But:

- Did the test really fail on master before your fix? Or was it a confirmation post-hoc?
- Is the function you patched actually reachable from a user-facing entry point, or is it dead code?
- Did your fix really cover the bug, or is there a boundary case where it still misbehaves?

`critic-orchestrator` spawns **three fresh `claude --print` subprocesses in parallel**, each with a hostile-by-design prompt:

| Worker | Role | Procedure |
|---|---|---|
| `falsification` | Pin the test | `git stash` → run test → `git stash pop` → run test. Test must fail pre-fix and pass post-fix. |
| `caller_verification` | Reach production | Grep call sites of the fixed function. At least one must be on a user-facing path. |
| `counterexample` | Break the fix | Brainstorm boundary cases; try to construct an input that still misbehaves. |

Each worker emits JSON-schema-constrained output. The orchestrator aggregates votes into a consensus: `claim_holds` / `claim_fails` / `split` / `undecided`.

**No Anthropic API key.** Workers inherit your Claude Code subscription via the `claude` CLI binary.

---

## Installation

```bash
git clone https://github.com/aureliocpr-ctrl/critic-orchestrator.git
cd critic-orchestrator
python -m pip install -e .[dev]
```

Dependencies: Python ≥ 3.12, `mcp`, `psutil`, plus the `claude` CLI on `$PATH`.

Run tests to confirm the environment:

```bash
python -m pytest tests -q
# 53 passed in ~2s
```

---

## MCP setup (Claude Code)

Register the server in `~/.claude.json`:

```json
{
  "mcpServers": {
    "critic-orchestrator": {
      "command": "python",
      "args": ["-m", "critic_orchestrator.mcp_server"],
      "env": {
        "PYTHONPATH": "/absolute/path/to/the/parent/of/critic_orchestrator",
        "PYTHONUTF8": "1"
      }
    }
  }
}
```

Restart Claude Code Desktop. Five tools become available:

| Tool | When to use |
|---|---|
| `start_adversarial_review` | **Default**. Returns a `job_id` in <100 ms; you poll for the result. |
| `poll_adversarial_review` | Check status / fetch the report. |
| `cancel_adversarial_review` | Kill a stuck or long-running review (process-tree, via psutil). |
| `list_adversarial_reviews` | Inspect running + recently completed jobs. |
| `force_adversarial_review` | **Legacy synchronous.** Blocks until done. Only safe if total wall < 60 s (i.e. counterexample-only). The MCP client times out at 60 s. |

See [USAGE.md](./USAGE.md) for the decision tree, full API contract, and operational patterns.

---

## Why the async-job pattern

The original `force_adversarial_review` was synchronous: the MCP tool call blocked until every worker finished. With the falsification worker (which runs `git stash + pytest + git stash pop + pytest`), end-to-end wall time routinely exceeded 60 s — and the Claude Code MCP client times out at 60 s with `-32001`, leaving the stdio of that session in a stale state until restart.

The async-job pattern fixes that:

```
start_adversarial_review(...)            ->  { "job_id": "abc..." }       (< 100 ms)
poll_adversarial_review(job_id)          ->  { "status": "running", ... }
poll_adversarial_review(job_id)          ->  { "status": "done", "result": {...} }
```

The worker subprocesses run inside a module-level `ThreadPoolExecutor(max_workers=8)`. `cancel_adversarial_review` kills the whole subprocess tree (claude → node) via `psutil`, not just the direct child.

---

## Hardening summary (2026-05-13)

After the initial async-job pattern was written, three adversarial reviews (two Code Reviewers + one Security Architect) plus the critic-orchestrator's own dogfooded run surfaced **4 HIGH + ~10 MED** issues. All HIGH and the economical MEDs are now fixed and pinned by tests:

| Issue | Fix |
|---|---|
| Cancel race during the spawn window leaked a real `claude` subprocess that ran to its full 180 s timeout | `Job.aborted` flag set BEFORE iterating Popen handles; `_spawn_worker` checks pre-spawn and post-register |
| `Popen.kill()` only signals the direct child; the `node.exe` grandchild kept grinding | `kill_process_tree` via `psutil.Process(pid).children(recursive=True)` |
| `claim` / `diff_summary` interpolated verbatim → prompt-injection vector with Bash-enabled worker | `_sanitize_for_prompt` (strip control chars, escape `</UNTRUSTED_INPUT>`, 4 KB clip) + envelope tags + security-boundary header |
| `_run_review_in_thread` swallowed `BaseException`, stranding the job on `running` forever | `except BaseException: mark_failed + raise` |
| GC ran only on `create` → memory leak in long-idle servers | GC also runs on `get`, `list_active`, `list_all` |
| `_EXECUTOR` never shut down on process exit | `atexit.register(lambda: _EXECUTOR.shutdown(wait=False, cancel_futures=True))` |

See [CHANGELOG.md](./CHANGELOG.md) for the full history.

---

## Smoke test (uses subscription tokens)

```bash
PYTHONPATH=/path/to/parent python smoke_live.py
```

Spawns ONE real `counterexample` worker against this repo's own code. Expected: `start` returns in < 5 ms, `done` in ~60 s, cost ~$0.30 of subscription tokens. Useful to confirm the MCP server and the `claude` CLI plumbing are wired up.

---

## License

All rights reserved. Private repository; not licensed for redistribution.
