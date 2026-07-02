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

Source-available: the code is public for inspection and learning.
© 2026 Aurelio Capriello. Not yet licensed for reuse or redistribution — an
open-source license is under consideration. If you'd like to use it, reach out.
