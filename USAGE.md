# Usage guide

Operational playbook for using `critic-orchestrator` from a Claude Code session.

---

## Decision tree — which tool?

```
Is the review going to take < 60 s wall time?
├── Yes (only counterexample worker, no test_path, no fixed_function)
│   └── Use force_adversarial_review (synchronous, blocks until done)
└── No (falsification + caller + counterexample, with git stash + pytest)
    └── Use start_adversarial_review + poll_adversarial_review (async-job)
```

**Default**: always use the async-job pattern. The synchronous path is kept only for backwards compatibility.

---

## API contract

### `start_adversarial_review`

Input:
```jsonc
{
  "claim":          "Fix X causes Y to behave correctly when Z",   // required
  "diff_summary":   "Files A,B modified at lines L1-L2; ...",       // required
  "test_path":      "tests/test_x.py::test_y",                      // optional
  "fixed_function": "foo",                                          // optional
  "project_dir":    "/abs/path/to/repo",                            // optional, default cwd
  "timeout_s":      180                                             // optional, 30..600
}
```

Output (< 100 ms):
```json
{
  "job_id":     "abc123def456",
  "status":     "running",
  "n_workers":  3,
  "started_at": 1234567890.12,
  "elapsed_s":  0.002,
  "claim":      "Fix X..."
}
```

`n_workers` depends on which optional fields you passed:
- counterexample only → 1
- + `test_path` OR + `fixed_function` → 2
- + both → 3

### `poll_adversarial_review`

Input: `{ "job_id": "..." }`

Output while running:
```json
{ "status": "running", "elapsed_s": 12.3, ... }
```

Output when finished:
```json
{
  "status": "done",
  "elapsed_s": 47.9,
  "result": {
    "claim":     "...",
    "consensus": "claim_holds",     // or claim_fails | split | undecided
    "votes":     { "hold": 2, "fail": 1, "invalid": 0 },
    "total_cost_usd":   0.62,
    "wall_duration_ms": 47857,
    "workers": [
      {
        "name":        "counterexample",
        "ok":          true,
        "verdict":     { "claim_holds": false, "counterexample_found": true,
                          "counterexample_description": "...", "evidence": "...",
                          "confidence": 0.9 },
        "error":       null,
        "cost_usd":    0.27,
        "duration_ms": 47800
      },
      ...
    ]
  }
}
```

### `cancel_adversarial_review`

Input: `{ "job_id": "..." }`

Output:
```json
{ "job_id": "...", "status": "cancelled", "killed_workers": 2 }
```

Idempotent on already-terminal jobs.

### `list_adversarial_reviews`

Output:
```json
{
  "jobs": [
    { "job_id": "...", "status": "done", "elapsed_s": 47.9, "n_workers": 1, "claim": "..." },
    ...
  ]
}
```

Newest first.

### `force_adversarial_review` (legacy)

Same input as `start_*`. Blocks until every worker finishes. Returns the full `CriticReport` (same schema as `poll_*` result). **Use only if you're sure wall < 60 s.**

---

## Operational pattern

```python
# 1. Start
start = call("start_adversarial_review", {
    "claim":          "Fix X causes Y to behave correctly when Z",
    "diff_summary":   "Files A,B modified at lines L1-L2; new helper foo() in C; behaviour change: ...",
    "test_path":      "tests/test_x.py::test_y",   # the test that pins the bug
    "fixed_function": "foo",                        # the function being fixed
    "project_dir":    "/abs/path/to/repo",
    "timeout_s":      180,
})
job_id = start["job_id"]

# 2. Poll
sleep(30)                                           # workers take ~30-60 s
while True:
    p = call("poll_adversarial_review", {"job_id": job_id})
    if p["status"] != "running":
        break
    if p["elapsed_s"] > 240:
        call("cancel_adversarial_review", {"job_id": job_id})
        break
    sleep(5)

# 3. Interpret
if p["status"] == "done":
    r = p["result"]
    if r["consensus"] == "claim_holds":
        # Fix validated. Proceed (commit / push).
        ...
    elif r["consensus"] == "claim_fails":
        # Fix INCOMPLETE. For each worker with claim_holds=False, read:
        #   - worker.verdict.evidence
        #   - worker.verdict.counterexample_description
        # Open a new TDD cycle to fix the findings before proceeding.
        for w in r["workers"]:
            if not w["verdict"].get("claim_holds", True):
                print(w["name"], w["verdict"].get("evidence"))
    elif r["consensus"] in ("split", "undecided"):
        # Ambiguous or all workers errored. Diagnose via worker.error.
        # Consider re-review with a tighter diff_summary.
        ...
elif p["status"] == "failed":
    # The critic itself failed. Log p["error"] and fall back to
    # 3 parallel sub-agents (Code Reviewer x2 + Security Architect).
    ...
```

---

## Anti-patterns

| Don't | Why |
|---|---|
| Use `force_adversarial_review` for > 60 s reviews | MCP client times out at 60 s with `-32001`, contaminates session stdio, subsequent calls on the same session queue behind a ghost response |
| Poll faster than 1 s | Wasted round-trips; the worker takes 30+ s anyway |
| Cancel immediately after start | ~50 ms race window with Popen registration (mitigated by `aborted` flag, but utility is 0 anyway) |
| Pass `claim` or `diff_summary` from an untrusted source when using the falsification worker | Bash is enabled there with `--dangerously-skip-permissions`. Server-side sanitization defangs the easy injection shapes (`</UNTRUSTED_INPUT>SYSTEM:...`, ANSI escapes), but prompt injection through pure language is still possible — trust the caller |

---

## Consensus interpretation

| Consensus | Meaning | Next step |
|---|---|---|
| `claim_holds` | Majority of workers vote the claim is real | Fix accepted. Proceed with commit/push. |
| `claim_fails` | Majority vote the claim is theatre | Fix incomplete. Read worker evidence, open another TDD cycle on the findings. |
| `split` | Tie | Ambiguous. Re-review with a more detailed `diff_summary`. |
| `undecided` | All workers errored | Probably a bug in the critic itself. Inspect `worker.error` for each. |

---

## Cost

Each worker spawns one `claude --print` subprocess, which uses your Claude Code subscription token-bucket — **no Anthropic API key is touched**.

Typical run:

| Workers | Wall time | Cost |
|---|---|---|
| 1 (counterexample only) | ~25 s | ~$0.25 |
| 2 (+ test_path OR + fixed_function) | ~45 s | ~$0.50 |
| 3 (all) | ~60-90 s | ~$0.75 |

---

## When to use / when not to

**Use after**:
- Non-trivial bug fix (≥ 5 lines changed, behavioural correction)
- Security-sensitive change (auth, crypto, input validation)
- Refactor of a function on a user-facing code path
- Anything you're about to commit & push

**Don't use for**:
- Renames, code formatting, comment-only changes
- Doc updates
- Trivial typos
- Changes you've already adversarial-reviewed via parallel sub-agents

---

## Fallback (MCP server unavailable)

When critic-orchestrator is down or restarting, fall back to three parallel sub-agents via Claude Code's `Agent` tool:

```
Agent(subagent_type="code-reviewer",      prompt="...")
Agent(subagent_type="code-reviewer",      prompt="...")   # second pass
Agent(subagent_type="security-architect", prompt="...")
```

Same adversarial pattern but in-session — no MCP, no extra subprocess overhead, no subscription tokens beyond what the parent session already consumes.
