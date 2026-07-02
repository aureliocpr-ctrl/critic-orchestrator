# Changelog

## 0.5.0 — 2026-07-02 — Ghost-CLI backend: full triad without `claude --print`

### Added
- `ghost_backend.py` — `CRITIC_BACKEND=ghost_cli`: each reviewer runs in a
  fresh **hidden interactive** Claude session ("ghost sister"), spawned
  `CREATE_NEW_CONSOLE` + `SW_HIDE` (invisible from birth) and driven via the
  `clp ai-eye` Win32 console transport. Long prompts travel by filesystem
  handshake (prompt file in, ONE short injected line, verdict JSON polled
  from a response file). The first alternative **agentic** backend: it runs
  `falsification` and `caller_verification` too — no `claude --print`, so
  reviews stay on the flat subscription when headless calls become metered.
  Technique ported from Engram's interactive judge (validated live
  2026-07-02, 10/10 decision agreement vs `claude -p`).
- Fresh sister per reviewer (adversarial independence: no shared session
  between falsification and counterexample), tree-killed in a `finally`.
- Lifecycle hardening: module-level live-sister registry with `atexit`
  sweep (an MCP server dying mid-review leaks no hidden `claude.exe`) and a
  hard cap `CRITIC_GHOST_MAX_SISTERS` (default 4) that fails fast with an
  honest error instead of piling invisible consoles onto the machine.
- Env: `CRITIC_WORKER_MODEL` (model pin, same var as the CLI path),
  `CRITIC_GHOST_BOOT_TIMEOUT` (default 60 s). Windows-only by declaration:
  `make_backend_from_env` raises a clear `ValueError` elsewhere.

### Verified live (2026-07-02)
- Execution reviewer (real `pytest` run) inside a ghost sister on a scratch
  project: correct verdict `{claim_holds: true, evidence: "1 passed in
  0.54s"}` via the response file, 31.4 s wall.
- True ghost: `MainWindowHandle == 0` at every 5 s sample while alive.
- No leak: zero new `claude.exe` processes after `run_worker` returned.

### Tests
- 75 → **93 passing** (+18: handshake contract, fenced/malformed/timeout
  responses, inject/start failures always close the session, cap fail-fast,
  slot release, registry thread-safety, boot trust-dialog handling, boot
  timeout with tail preview, env selection incl. non-Windows rejection).

---

## 0.4.0 — 2026-07-01 — Multi-provider backends

### Added
- Provider backends (`backends.py`): run the reviewers through the Claude
  CLI (agentic — all workers) OR any OpenAI-compatible API / the Anthropic
  API (reasoning-only). Selected via `CRITIC_BACKEND`
  (`claude_cli` | `openai_compat` | `anthropic_api`) + `CRITIC_MODEL`.
  No new dependency — the API backends use `urllib`.
- Honest capability split: reviewers that need real tools
  (`falsification`, `caller_verification`) carry `requires_execution=True`
  and are reported `skipped: requires an agentic backend` on API backends —
  never faked. `counterexample` (reasoning) runs on any provider.
- Per-provider structured-output negotiation (`CRITIC_JSON_MODE=auto`,
  default): tries `json_schema` → `json_object` → none, falling back only
  on the specific "response_format unsupported" 400.
- Temperature is omitted unless `CRITIC_TEMPERATURE` is set.

### Verified live (2026-07-01)
- Moonshot `kimi-k2.7-code`: accepts `json_schema`; rejects `temperature != 1`
  (hence temperature omitted by default). Reasoning reviewer returned a
  correct, schema-conforming verdict (~15 s).
- DeepSeek `deepseek-v4-pro`: rejects `json_schema` → auto-fell-back to
  `json_object`; reasoning reviewer returned a correct verdict (~9 s).

### Tests
- 56 → **75 passing** (+19: backend unit tests incl. format-fallback,
  temperature omission, and env-driven selection).

---

## 0.3.0 — 2026-05-15 — Counterexample prompt debiasing (empirical)

### TL;DR

The counterexample worker prompt had a single line —
`"Bias hard toward FINDING counterexamples. A 'no counterexample'
answer should be the conclusion of work, not a default"` — that
drove the **false-positive rate to 100%** in a controlled
experiment. Removing it and adding an explicit false-positive check
brings the FP rate to **0%** with no measurable cost in recall.
This release ships only the prompt change. No architectural change.

### The experiment that drove this release

Two symmetric experiments under `experiments/`:

**1. `exp_variance_bias.py`** — ground truth = `claim_holds=TRUE`
(feature really implemented in `mcp_server.py:71`). N=3 baseline +
N=3 debiased. Measures **false-positive rate**.

| Condition | claim_holds=TRUE | counterexample_found=False | FP rate |
|---|---|---|---|
| baseline (v0.2.0 prompt with "Bias hard toward FINDING") | 0/3 | 0/3 | **3/3 = 100%** |
| debiased (no "Bias hard", + anti-FP check) | 3/3 | 3/3 | **0/3 = 0%** |

The 3 baseline false positives were all inventions: a UNC path the
fix never claimed to handle, a 50 MB claim string (sanitizer clips
at 4 KB), `Path.resolve()` "slow" on Windows (it's millisecond).
Each fell under at least one Anthropic-documented false-positive
pattern.

**2. `exp_bug_injection.py`** — ground truth = `claim_holds=FALSE`
(temporary controlled bug: replaced `raise` after `mark_failed` with
`return` at `mcp_server.py:268`, then reverted). Measures **false-
negative rate**.

| Condition | claim_holds=FALSE | counterexample_found=True | FN rate |
|---|---|---|---|
| baseline | 3/3 | 3/3 | **0/3 = 0%** |
| debiased | 3/3 | 3/3 | **0/3 = 0%** |

Both prompts caught the injected bug 3/3 times. Recall is preserved.

**Confidence is uncorrelated with correctness.** Across both
experiments the worker's self-reported `confidence` sat in the
0.85-0.99 band whether the verdict was correct (TP, TN) or wrong
(FP). A confidence-threshold filter (à la Anthropic's
`code-review.md` step 5-6) would have rejected zero of the FPs we
saw. We do NOT ship confidence-based filtering.

**Limit of the experiment**: only one ground-truth-FALSE case, and
the injected bug was obvious (a one-line `return` vs `raise`). The
test does not yet probe whether the debiased prompt misses *subtle*
bugs. This is the known weakness of v0.3.0; addressing it requires
a richer ground-truth corpus.

### Changed

- `default_workers.py::_counterexample_worker` prompt rewritten:
  - Opens with "evaluate whether the fix is sound — OR honestly
    conclude none exists", instead of "break the fix".
  - Both `counterexample_found=True` and `counterexample_found=False`
    explicitly marked as equally valid outcomes.
  - Added the slogan "Confabulated bugs are worse than missed bugs".
  - 7-bullet "FALSE-POSITIVE CHECK" section adapted from the
    Anthropic code-review plugin's list of FP patterns:
    pre-existing issues, hypothetical-version bugs, pedantic
    nitpicks, linter-catchable issues, lines outside diff,
    general code-quality issues, environmental preconditions.
- `tests/test_hardening.py` gains 3 pinned tests so a future
  regression on the prompt is caught by CI:
  - `test_counterexample_prompt_does_not_force_finding`
  - `test_counterexample_prompt_legitimises_no_counterexample`
  - `test_counterexample_prompt_has_false_positive_check`

### Added

- `experiments/exp_variance_bias.py` — FP-rate harness.
- `experiments/exp_bug_injection.py` — FN-rate harness.
- `experiments/results_variance_bias.json` — raw run data (6 runs).
- `experiments/results_bug_injection.json` — raw run data (6 runs).

### Not changed (deliberately)

Three modifications that were *suggested* by independent reviewers
or by an earlier reading of the Anthropic code-review plugin —
**none of them survived the empirical test**:

- **Confidence-weighted voting.** Suggested by an earlier review.
  Smoke data shows confidence is identical for correct and wrong
  verdicts (0.88 vs 0.87 across our 12 runs). Not shipped.
- **Ensemble / N-of-M majority voting.** Suggested as a cure for
  LLM noise. Smoke data shows zero within-condition variance
  (3/3 identical verdicts in every cell) — the worker is not
  noisy, it is *systematically* biased by the prompt. Ensembles
  do not help. Not shipped.
- **Second-stage scoring agent** (Anthropic-style "Haiku scores
  the score"). Cost ~+50% per review; the bias is upstream of the
  scoring step, so a second pass on the same biased verdict
  would not necessarily decorrelate. Deferred until we have data
  showing it helps.

### Total test count

53 → **56 passing in ~1.7s**.

### Cost of the experiments that drove this release

12 real `claude --print` worker runs, **$5.03 of subscription
tokens** (no Anthropic API key touched). Receipts in
`experiments/results_*.json`.

---

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
