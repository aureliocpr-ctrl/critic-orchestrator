# Changelog

## Unreleased — The three wirings: execution reaches third-party providers

The remaining residue was three instances of the same class this product
hunts — *built-never-wired* — and closing them was gated on its own grid.

### Added
- **Pinned execution wired into `falsification`** (`WorkerSpec.exec_request`,
  `AgenticApiBackend.exec_policy`): with the operator opt-in
  (`CRITIC_ALLOW_EXEC=1` + `CRITIC_EXEC_ROOTS`), the API backends run the
  falsification reviewer through ONE no-argument tool,
  `run_falsification_experiment`, that executes the experiment code already
  decided: the test at HEAD in an ephemeral worktree (expected PASS) and at
  the pre-fix baseline with only the test file taken from HEAD (expected
  FAIL). The `exec` capability is decided **per run** — it exists only for a
  `project_dir` the operator allowlisted; a denial travels into the skip
  message with the knobs named. Third-party providers go from 2/3 to **3/3
  reviewers — verified live on DeepSeek** reviewing this repository's own
  editable-install fix (falsification 0.98, explicitly distinguishing the
  pinned assertion's failure from an import error).
- **`start_product_probe` (MCP)** — the promises finally have an executor:
  extraction (already shipped) now feeds `run_promise`/`run_product_probe`,
  which execute each promise in a worktree at HEAD (argv via shlex, no
  shell, per-promise timeout with tree kill, cancellable between promises)
  and score kept/broken/not_run. Fail-fast at start when the operator
  opt-in is missing. `_REFUSED_PATTERNS` learns the **shell-reintroduction**
  class (`powershell`, `pwsh`, `cmd`, `bash`/`sh`/`zsh`/`fish`/`ksh`,
  `env`, `xargs`, `eval`, `exec`, `start`, `format`, `fdisk`, `mount`) —
  refused at extraction AND re-refused at execution.
- **Ghost backend cancellation**: `cancel_check` pre-flight (no sister boot
  for an already-cancelled review), post-boot, and per-tick in the response
  poll; the one blind window (the boot, bounded by `boot_timeout_s`) is
  documented. Previously a cancelled ghost review ran to its full timeout
  with the hidden sister alive.

### Fixed
- **The probe out-imports an editable install of the reviewed package.**
  Found by refusing to trust the first green: an editable install resolves
  imports to the REAL directory (at HEAD, fix present), so the pre-fix run
  would import post-fix code, pass, and the probe would report
  "confirmation post-hoc" about a genuine falsification — a confident wrong
  verdict about the reviewer's own method. Cure: the worktree checkout is
  nested at `<container>/<repo-name>` and every probe run gets
  `PYTHONPATH=[container, worktree, worktree/src]` ahead of the inherited
  value, which precedes site-packages where both editable mechanisms
  resolve. The first pinning test used a layout where the nested checkout
  alone already won (tests *inside* the package) and the mutation survived;
  rewritten against `tests/` without `__init__.py`, where PYTHONPATH is the
  only defence.
- **`fs_grep` streams instead of slurping.** A live grep over a directory
  containing session logs died with `MemoryError`: `read_text` loads whole
  files, and a jsonl log can be ONE multi-hundred-MB line. Now line-wise
  `readline(cap)` with per-file (2 MB) and per-line (64 KB) caps, exact
  line numbers for normal files, partially-scanned files named in the
  result.

### Validation
- 68 new tests (suite 349 passed, 2 skipped, ruff clean repo-wide).
- 14 targeted mutations across the three wirings, each killed by its
  pinning test (grant-never-built, cache-removed, frame-removed,
  pre-runs-on-head, exec-request-unwired, pythonpath-not-set,
  src-layout-dropped, preflight-removed, poll-check-removed,
  policy-not-consulted, re-refusal-removed, extract-from-real-tree,
  cancel-ignored, fail-fast-removed); restores verified after every run.
- Live: DeepSeek 3/3 on the post-fix triad with execution
  (`experiments/exp_falsification_live.py`); the first two runs of that
  experiment failed honestly — a wrong `project_dir` produced the policy
  denial exactly as worded, and the harness read `report` where the tool
  returns `result` — both fixed in the experiment, not papered over.

## Unreleased — Design review: the no-claim gate + deterministic audit

### Added
- `design_workers.py` + MCP tool `start_design_review` — adversarial review of
  **modules or a design doc, before/without a fix**, that accepts **no claim and
  no diff** (enforced by the tool schema and pinned by a signature test). Three
  read-only reviewers infer the module's purpose from the artifact and review it
  through independent lenses: `premortem` (Klein prospective hindsight with an
  honest empty-handed exit), `perimeter` (category errors / is-this-the-right-
  problem, frame attack allowed), `detection` (FMEA × STAMP: can each safeguard
  fire, who notices its silent death).
  - **Why it exists:** the post-fix reviewers take the proposer's claim, and the
    claim defines the review perimeter — measured as four consecutive 3-0
    `claim_holds` votes on changes mutation testing later showed to contain five
    theatre-tests. Removing the claim is the structural cure; a harsher prompt is
    not (that was v0.2.0, empirically 100% false positives).
  - A historical failure-class grid is injected as an **additive** search aid
    ("EXTEND your search; never limit it" — contractual, pinned by test).
  - Aggregation by deduped, severity-ranked **findings** (`DesignReport`,
    duck-typed into the existing async job registry via a `ReviewReport`
    Protocol); cross-lens dedupe on file+line with corroboration and
    `also_reported_as` recorded, plus `findings_per_file` so one root cause
    reported by three lenses does not read as three problems.
- `audit_detectors.py` (+ `python -m critic_orchestrator.audit_detectors`) — two
  deterministic, LLM-free detectors that feed the grid: **dead env flags**
  (built-never-wired: falsy-default flag nothing sets — `or`-fallback and
  documented/config/test tiers handled) and a **deviation register** (declared
  limitations with git-blame age; no silent caps — truncation is reported).

### Validation (numbers restated after an adversarial re-review — see Fixed)
- First live run (3 no-claim reviewers × Opus on a 5.7k-line retrieval core):
  11 findings verified line-by-line, **0 false positives** — **7 genuinely new
  defects** + 4 the code already declared as known limits. Among the new ones: a
  lock omission on the exact breaker path a claim-fed review had voted 3-0 clean
  the day before, a default-ON fusion path bypassing the `valid_until` gate its
  own docstring promises "in every mode", and a deserializer silently dropping a
  provenance column on every read.
  - The first write-up said "10 new": it counted one defect twice (two lenses,
    two angles) and promoted two code-declared limits to "new". Restated.
- Capture effect: the arm given the proposer's *true* claim put 3/3 findings
  inside that claim's subject vs 1/5 for the same lens+grid without it.
  Predicted direction, **not significant** (Fisher exact p=0.14, n=1 per arm).
  - The first write-up compared 3/3 against 0/3 — an arm differing in *two*
    factors (claim AND grid). The single-factor comparison is 3/3 vs 1/5.
- Reviewers ran on the proposer's own model family: this validates the **input
  contract**, not model independence. Harness: `experiments/exp_design_review.py`.
- Suite: 138 passed (was 111).
- pyproject version intentionally NOT bumped: committed, not released —
  features accumulate to one tag.

### Added — `agentic_api`: every reviewer, on any tool-calling provider
- `agentic_api.py` + `CRITIC_BACKEND=agentic_api` — an agent loop over the
  same OpenAI-compatible endpoints, with read-only sandboxed `fs_read /
  fs_list / fs_glob / fs_grep`, so the design lenses run on Kimi, GLM,
  DeepSeek and friends. `urllib` only, no new dependency, and no dependency
  on any external coding-agent CLI (this package must stay usable by
  someone who has none).
- **Capability model.** `WorkerSpec.needs` (`read` / `exec`) vs a backend's
  `capabilities`; needs beyond the offer means SKIPPED before any request.
  Necessary because `requires_execution` could not tell
  `caller_verification` (grep is its whole method) from `falsification`
  (whose method IS `git stash` + run the test) — the latter, left running
  on a read-only sandbox, would have reasoned its way to
  `test_falsifies_master: true` having executed nothing.
- **Convergence.** A step trace in `raw_preview` on every error path, a
  wind-down notice two steps from the end, and `tool_choice:
  submit_verdict` forced on the final step (retried without it on
  providers that reject it).
- Live: **DeepSeek PASS** (5 findings, 133 s) · **GLM 4.6 PASS** after the
  convergence fix (2 findings, 186 s; it had burned all 14 steps without
  submitting) · Kimi k3 `ConnectionResetError` — transport, reported
  without a fabricated verdict. Backend table in the README: agentic_api
  is 2 of 3 on the post-fix triad, 3 of 3 on the design lenses.
- Independent models earned their keep immediately: DeepSeek found a
  **critical, true** defect in this repo's own cancellation path (see
  Fixed), and GLM's flag on TTL retention exposed a real inconsistency
  (`mark_cancelled` was the only terminal transition without a sweep —
  and the one a caller reaches without polling afterwards). GLM's second
  finding was FALSE (it claimed no explicit registry reset in tests;
  four exist) — it had been shown one file, the blind-judge failure mode.
- Endpoint construction no longer invents a version segment: GLM lives
  under `/api/paas/v4` and an unconditional `/v1` produced
  `/v4/v1/chat/completions` and a live 404. Eight parametrised cases.

### Fixed — found by turning this tool's own three lenses on itself
- **`design_review()` had zero call sites** (product and tests): dead code, the
  built-never-wired class this tool's grid leads with. Now the verified
  library-facing API with an end-to-end test.
- **`audit_detectors` was reachable only from its own tests** — no MCP tool, so
  the "audit gate" an agent could invoke did not exist. Now `run_repo_audit`.
- **Default per-worker timeout was 300 s while 5 of the 9 measured design
  workers ran 260-408 s** — the shipped default would have timed out the
  majority of the run that validated the tool. Now 600 s (max 900), with the
  measurement in the docstring. Same class as the daemon-lease constant a review
  caught in the audited repo: a number true in its regime, false in the one it
  must cover.
- **`find_deviations` stashed its truncation flag on the function object** —
  process-global mutable state under an 8-thread executor, so two concurrent
  audits could overwrite each other's flag and the anti-silent-cap mechanism
  could itself go silent (state-leak-across-runs, from this tool's own grid).
  Truncation is now returned; a concurrency test pins it.
- **A review that lost lenses to timeouts reported `no_blocking_findings`** — a
  reassuring verdict derived from a third of the review (silent-failopen, also
  from the grid). New `incomplete` status plus `lenses_ok` / `lenses_failed` in
  every report; a blocking finding still wins over incompleteness.

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
