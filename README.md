# critic-orchestrator

> An anti-confabulation verification layer for AI coding agents. It spawns
> independent adversarial reviewers, in parallel, to check a claimed
> fix — *before* you trust it. Provider- and agent-agnostic.

**Status**: 75/75 tests green. Used daily in my own workflow; APIs beyond the
Claude CLI backend are unit-tested and live-smoke-tested (see *Verified* below).

---

## The problem

When an AI coding assistant tells you *"I fixed the bug, the test passes now"*,
three things are easy to miss:

- Did the test actually **fail before** the fix, or was it written to pass
  after the fact (a confirmation, not a real regression test)?
- Is the function that was patched actually **reached** from a real entry
  point, or is it dead code?
- Does the fix hold at the **boundaries**, or is there an input where the
  bug still bites?

These are exactly the gaps that produce confident-but-wrong output. This tool
turns those three questions into three independent checks.

---

## How it works

`critic-orchestrator` runs **three adversarial reviewers in parallel**, each
starting from a fresh context and prompted to *disprove* the claim, not agree
with it:

| Reviewer | Role | Procedure |
|---|---|---|
| `falsification` | Pin the test | `git stash` → run the test → `git stash pop` → run again. The test must **fail** before the fix and **pass** after. |
| `caller_verification` | Reach reality | Find the call sites of the patched function. At least one must sit on a real entry point (CLI, API, hook), not only tests. |
| `counterexample` | Break the fix | Look for a concrete input/boundary case where the fixed code still misbehaves. |

Each reviewer returns a JSON-schema-constrained verdict; the orchestrator
aggregates them into a consensus: `claim_holds` / `claim_fails` / `split` /
`undecided`.

The reviewers run in an async job model: `start_adversarial_review` returns a
`job_id` in under 100 ms, then you poll — so a long review never trips the
60 s MCP client timeout.

---

## Before the fix: design review (the no-claim gate)

The three reviewers above verify a *claim* against a *diff* — they run
**after** a change. That contract has a measured failure mode: the proposer
writes the claim, and the claim silently defines the review perimeter (the
counterexample reviewer is instructed to discard findings outside the claim).
The review then scores the proposer's prudence, not the artifact — observed
here as four consecutive 3-0 `claim_holds` votes on changes where deterministic
mutation testing later found five theatre-tests.

`start_design_review` is the structural cure. It reviews **modules or a design
doc, before or without a fix**, and it accepts **no claim and no diff** — by MCP
schema, not by convention. The reviewers read the files raw and infer what the
code is *for* from the artifact itself:

| Reviewer | Lens | Human practice it encodes |
|---|---|---|
| `premortem` | "It **has** failed — reconstruct the post-mortem." | Klein's prospective hindsight (measurably out-produces "please critique this"), with an honest empty-handed exit. |
| `perimeter` | "Is this the **right problem**?" Category errors, not tuning. | Murder-board / red-team: the reviewer may attack the frame, not only the build. |
| `detection` | "Can each declared safeguard **fire**, and who notices its **silent death**?" | FMEA-detection × STAMP control-loop audit. |

A grid of historically frequent failure classes (built-never-wired,
guard-that-does-not-guard, test-theatre, silent-failopen, …) is injected as an
**additive** search aid — it extends the reviewers' search, it never limits it.
Output is aggregated by deduped, severity-ranked **findings** (a design review
has no binary claim to hold or fail), with cross-lens corroboration recorded.

On a first live run against a 5.7k-line retrieval core, the three no-claim
reviewers produced 11 findings that survived line-by-line verification with **0
false positives**: **7 genuinely new defects** (including a lock omission on the
exact path a claim-fed review had voted 3-0 clean the day before) and 4 that the
code itself already declared as known limits. A control arm given the proposer's
*true* claim put 3 of 3 findings inside that claim's subject, against 1 of 5 for
the same lens without it — the capture effect this contract removes, in the
predicted direction but **not statistically significant** at n=1 per arm
(Fisher exact p=0.14). Single run, and the reviewers ran on the same model
family as the proposer: this measures the *input contract*, not model
independence.

### Backends: design review needs an *agentic* one

All three design lenses read files (`Read`/`Grep`/`Glob`), so they carry
`requires_execution=True`. Consequence, stated plainly because it is easy to
miss: on a **reasoning-only** API backend (`openai_compat`, `anthropic_api`) a
design review runs **zero of three** reviewers — they are honestly reported as
`skipped: requires an agentic backend` and the report comes back `undecided`,
never fabricated. That is stricter than the post-fix triad, where the
`counterexample` reviewer still runs on any provider (1 of 3).

| Backend | offers | post-fix triad | design review |
|---|---|---|---|
| `claude_cli` / `ghost_cli` (subscription) | read + exec | 3 of 3 | **3 of 3** |
| **`agentic_api`** (Kimi, GLM, DeepSeek, …) | read | **2 of 3** | **3 of 3** |
| `openai_compat` / `anthropic_api` | — | 1 of 3 | 0 of 3 |

`agentic_api` drives those same OpenAI-compatible endpoints as a real agent loop
with read-only filesystem tools, so the design lenses run on any provider with
tool calling — no coding-agent CLI in the middle, no extra dependency.

**Why 2 of 3 and not 3.** Each reviewer declares what it *needs*
(`WorkerSpec.needs`) and each backend declares what it *offers*
(`capabilities`); a reviewer whose needs exceed the offer is skipped, never
answered. `caller_verification` needs `read` (grep is its whole method) and
runs. `falsification` needs `exec` — its method *is* `git stash` + run the test
+ restore + run again — so a read-only sandbox refuses it. That refusal is the
point: left to run, the model would read the test, *reason* about whether it
would fail pre-fix, and answer `test_falsifies_master: true` having executed
nothing. A conclusion with no observation behind it is exactly the
confabulation this tool exists to catch, so no verdict is better than that one.

### Deterministic audit (`audit_detectors.py`)

Some failure classes are bookkeeping, not judgment, and an LLM only adds
confabulation risk. Two grep-based detectors (no model, no network) feed the
design grid:

- **dead env flags** — capability gated behind an env flag defaulting OFF that
  no config, doc, script, or assignment ever sets (the built-never-wired class).
- **deviation register** — declared limitations (`KNOWN LIMIT`, `for now`,
  `FIXME`) collected *with their git-blame age*, so an old unrevisited deroga
  becomes a queue item instead of wallpaper. No silent caps: truncation is
  reported.

Exposed as the MCP tool **`run_repo_audit`** (returns immediately, no job), or
standalone: `python -m critic_orchestrator.audit_detectors <repo> --no-age`.

The detectors report *candidates* for a judge, not verdicts, and their blind
spots are declared: the flag detector models `os.environ.get()` / `os.getenv()`
(a subscript occurrence anywhere counts as a reference, so a blind spot never
reads as evidence of deadness), and it treats an `or`-fallback as the effective
default — a false positive found on the first live run and pinned by a test.

---

## Works with any coding agent

The tool is an **MCP server**, so any MCP-capable coding agent can call it
(Claude Code today; any client that speaks the Model Context Protocol). It is
a callable primitive, not a plugin tied to one host.

## Works with any provider

Each reviewer is run by a **backend**, selected via environment variables.
There is a hard, honest capability split:

| Backend | `CRITIC_BACKEND` | Can run | Notes |
|---|---|---|---|
| Claude CLI (agentic) | `claude_cli` (default) | **all 3 reviewers** | Has real Bash/Grep/Read tools, so it can run the test and grep call sites. Uses your Claude Code subscription — no API key. |
| Ghost CLI (agentic) | `ghost_cli` | **all 3 reviewers** | Windows-only. Each reviewer runs in a fresh **hidden interactive** Claude session (no `claude --print` anywhere), driven via the `clp ai-eye` console transport with a filesystem handshake. Stays on the flat subscription when headless calls become metered. |
| Any OpenAI-compatible API | `openai_compat` | `counterexample` (reasoning) | OpenAI, DeepSeek, Moonshot/Kimi, OpenRouter, Groq, local vLLM/Ollama. One shot, JSON-schema output. |
| Anthropic API | `anthropic_api` | `counterexample` (reasoning) | Native Messages API, structured output. |

**Why the split is honest, not a limitation hidden:** a plain chat-completion
API makes a single call — it cannot run `git stash`, `pytest`, or `grep`. So
the two reviewers that *require* executing commands are reported as
`skipped: requires an agentic backend` — **never faked**. The reasoning
reviewer (`counterexample`) runs on any provider. To run the full triad on a
non-Claude provider you need an *agentic* backend (an agent CLI with tools);
that adapter is a documented, in-progress extension point, and providers differ
in how they expose tool-calling — so it is added per-provider, verified against
the real endpoint, not guessed.

```bash
# Example: reasoning reviewer via DeepSeek
export CRITIC_BACKEND=openai_compat
export CRITIC_BASE_URL=https://api.deepseek.com
export CRITIC_MODEL=deepseek-v4-pro
export CRITIC_API_KEY=...        # your key
# structured output is auto-negotiated (json_schema → json_object → none)

# Example: via Anthropic API
export CRITIC_BACKEND=anthropic_api
export ANTHROPIC_API_KEY=...
export CRITIC_MODEL=claude-sonnet-5

# Example: full triad WITHOUT claude --print (Windows + clp arsenal)
export CRITIC_BACKEND=ghost_cli
export CRITIC_WORKER_MODEL=opus            # model pin for the sisters
export CRITIC_GHOST_MAX_SISTERS=4          # hard cap on hidden sessions
```

Temperature is not sent unless you set `CRITIC_TEMPERATURE` — some reasoning
models reject any value other than their default.

---

## Install

```bash
git clone https://github.com/aureliocpr-ctrl/critic-orchestrator.git
cd critic-orchestrator
python -m pip install -e .[dev]
python -m pytest tests -q        # 93 passed in ~3s
```

Requires Python ≥ 3.12 and `psutil`. The default Claude CLI backend also needs
the `claude` CLI on `$PATH`; the API backends use only the standard library.

## MCP setup

Register in `~/.claude.json` (or your agent's MCP config):

```json
{
  "mcpServers": {
    "critic-orchestrator": {
      "command": "python",
      "args": ["-m", "critic_orchestrator.mcp_server"],
      "env": {
        "PYTHONPATH": "/absolute/path/to/parent/of/critic_orchestrator",
        "PYTHONUTF8": "1"
      }
    }
  }
}
```

Five tools become available:

| Tool | When |
|---|---|
| `start_adversarial_review` | **Default.** Returns a `job_id`; you poll. |
| `poll_adversarial_review` | Check status / fetch the report. |
| `cancel_adversarial_review` | Abort a running review (kills the whole worker process tree). |
| `list_adversarial_reviews` | Inspect running + recent jobs. |
| `force_adversarial_review` | Legacy synchronous path; safe only for sub-60 s reviews. |

See [USAGE.md](./USAGE.md) for the full API contract and operational patterns.

---

## Caveats — read before enabling the falsification reviewer

- The `falsification` reviewer runs `git stash` and `git stash pop` on the
  working tree to test with and without the fix. **Do not run it while you
  have uncommitted edits in flight**, and prefer running it on a clean commit:
  a concurrent edit can collide with the stash. Committing first is the safe
  pattern.
- The falsification reviewer executes commands (test runner) with tool access
  pre-authorized (non-interactive mode has no prompt to confirm). Its tools are
  restricted by an allow-list, but treat it like any tool-enabled agent: run it
  on code you trust.

## Untrusted input handling

The `claim` and `diff_summary` you pass in are wrapped in explicit
data-only envelopes and sanitized (control characters stripped, envelope-escape
sequences neutralized, clipped to 4 KB) before reaching a reviewer. This is
defense-in-depth for the case where those fields originate from an untrusted
source (a commit message, a CI artifact). It is not a substitute for trusting
the caller — it removes the easy injection shapes, not the need for judgment.

---

## A note on prompt design (empirical)

The `counterexample` reviewer once carried a single instruction —
*"bias hard toward finding a counterexample"* — that pushed its false-positive
rate to **100%** in a controlled experiment (it invented problems that didn't
exist). Removing that line and adding an explicit false-positive checklist
brought it to **0%** with no measurable loss in real-bug recall. The raw runs
are in [`experiments/`](./experiments/) and the reasoning is in the
[CHANGELOG](./CHANGELOG.md).

The design principle it left behind: *a confabulated bug is worse than a missed
one.* The tool that checks for confabulation must hold itself to the same
standard — including not shipping "confidence-weighted voting" or "ensembles"
that the data showed didn't help.

---

## Verified

- Test suite: **93/93 green** in ~3 s (`python -m pytest tests -q`).
- Claude CLI backend: end-to-end smoke via `smoke_live.py`.
- Ghost CLI backend: **live smoke 2026-07-02** — an execution reviewer ran a
  real `pytest` inside a hidden sister (verdict + verbatim pytest summary via
  the response file, 31.4 s wall), `MainWindowHandle == 0` on every sample
  while alive (true ghost), zero `claude.exe` left after the run (tree-kill).
- OpenAI-compatible backend: unit-tested (mocked transport) and **live
  smoke-tested against two providers with different quirks** — Moonshot Kimi
  `k2.7-code` (accepts `json_schema`, rejects `temperature != 1`) and DeepSeek
  `deepseek-v4-pro` (rejects `json_schema`, auto-falls-back to `json_object`).
  In both, the reasoning reviewer returned a correct verdict.
- Anthropic API backend: unit-tested (mocked transport); live use needs your key.

---

## License

**MIT** © 2026 Aurelio Capriello. See [LICENSE](./LICENSE). Free to use, modify,
and redistribute — attribution appreciated.
