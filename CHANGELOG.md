# Changelog

## 0.2.0 — 2026-05-13 — Async-job pattern + hardening

### Added

- **Async-job pattern**: 4 new MCP tools (`start_adversarial_review`,
  `poll_adversarial_review`, `cancel_adversarial_review`,
  `list_adversarial_reviews`). `start` returns a `job_id` in < 100 ms; the
  caller polls for the result. Bypasses the 60 s MCP client timeout that
  killed every multi-worker review.
- `JobRegistry` (`job_registry.py`): thread-safe `RLock`-guarded in-memory
  store with TTL-based GC. GC fires on every read path (`create`, `get`,
  `list_active`, `list_all`) — no longer starves when the server idles.
- `Job.aborted` flag — set by `cancel` **before** iterating Popen handles
  so worker threads spawning a new subprocess observe it on their pre/post
  Popen checks and abort cleanly instead of racing into a leaked process.
- `kill_process_tree` (`orchestrator.py`): walks
  `psutil.Process(pid).children(recursive=True)` and kills each before
  killing the parent. Fixes the bug where `Popen.kill()` only signals the
  direct `claude.exe` child, leaving its `node.exe` runtime orphaned for
  the full timeout. Surfaced by the critic-orchestrator's own counterexample
  worker during adversarial review of this fix (confidence 0.92).
- `cancel_check` callable parameter on `_spawn_worker` and
  `adversarial_review`. Three checkpoints per worker: pre-Popen, post-Popen,
  post-register. Closes the cancel race window completely.
- `_sanitize_for_prompt` (`default_workers.py`): strips C0/C1 control
  characters (ANSI escapes, NULs), escapes `</UNTRUSTED_INPUT>` and
  `<UNTRUSTED_INPUT` open-tag attempts, clips to 4 KB. Every interpolated
  `claim`/`diff_summary`/`test_path`/`fixed_function` is now wrapped in
  `<UNTRUSTED_INPUT type="...">` envelope tags with an explicit
  security-boundary header that instructs the worker to treat the
  contents as data, not directives.
- `_EXECUTOR`: dedicated module-level `ThreadPoolExecutor(max_workers=8)`.
  Background reviews survive `asyncio.run()` cycles that would otherwise
  shut down the default executor.
- `atexit.register(lambda: _EXECUTOR.shutdown(wait=False, cancel_futures=True))`
  — best-effort cleanup so SIGINT does not hang the MCP server on stuck
  in-flight reviews.
- Test suite: 53 tests across 4 files (`test_job_registry.py`,
  `test_orchestrator_popen.py`, `test_mcp_async_tools.py`,
  `test_hardening.py`). Runs in ~2 s. Covers lifecycle, cancellation race
  windows, process-tree kill, prompt-injection sanitization, GC paths,
  and BaseException propagation.
- `smoke_live.py`: end-to-end smoke test against a real `claude --print`
  worker. Uses subscription tokens. Useful to verify MCP wiring.
- `README.md`, `USAGE.md`, `CHANGELOG.md`, `pyproject.toml`, `.gitignore`.

### Changed

- `_spawn_worker` (`orchestrator.py`) rewritten from `subprocess.run` to
  `subprocess.Popen` + `communicate(timeout=...)`. Required to expose
  the handle for external cancellation. Adds a completion log entry
  (`END: state=... duration_ms=... cost_usd=...`) to the per-worker file
  in `%TEMP%/critic_orchestrator/`.
- `_run_review_in_thread`: catches `BaseException` (not just `Exception`)
  to ensure a `KeyboardInterrupt` / `SystemExit` / `CancelledError` marks
  the job `failed` and re-raises — previously, those would silently absorb
  the signal and strand the job on `running` forever.
- `force_adversarial_review` description in MCP tool listing now starts
  with `[LEGACY SYNCHRONOUS]` and recommends the async path.

### Security

- Defangs prompt-injection vectors through the `claim` and `diff_summary`
  fields by stripping control characters, escaping the `</UNTRUSTED_INPUT>`
  envelope-escape vector, and wrapping interpolated text in explicit
  data-only envelopes. Risk surfaces when the falsification worker is
  enabled (Bash allowed, `--dangerously-skip-permissions`); the
  hardening is defense-in-depth, not a substitute for trusting the
  MCP caller.

### Adversarial review of this release

Three independent reviewers + one dogfooded self-review run:

| Reviewer | Findings |
|---|---|
| Code Reviewer #1 | 1 HIGH (cancel race) + 2 MED (GC, executor queue) + 3 LOW |
| Code Reviewer #2 | 0 HIGH, 6 MED, 4 LOW |
| Security Architect | 2 HIGH (prompt injection, cancel race) + 3 MED + 3 LOW |
| critic-orchestrator (live, counterexample) | 1 HIGH (process-tree kill) |

All HIGH issues and economical MEDs are fixed and pinned by tests.

### Verified

- Suite: 53/53 green in ~2 s.
- Smoke #1 (baseline async-job): `start` 1.7 ms, `done` in 47.8 s, $0.27.
- Smoke #2 (post-hardening): `start` 2.8 ms, `done` in 63.9 s, $0.29.
- 5 tools listed correctly by the MCP server after a clean restart.

---

## 0.1.0 — 2026-05-11 — Initial release

- `force_adversarial_review` MCP tool (synchronous).
- Three default workers: `falsification`, `caller_verification`,
  `counterexample`.
- `ThreadPoolExecutor`-based parallel spawn.
- `--strict-mcp-config` + `--dangerously-skip-permissions` worker
  invocation.
- `stdin=DEVNULL` to prevent the worker from inheriting the MCP
  JSON-RPC pipe and hanging on a `--print` stdin fallback.
