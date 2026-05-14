"""Empirical experiment #2 — false negative rate on injected bug.

GROUND TRUTH: The claim below describes a guarantee that the code USED
to have (the `raise` after `mark_failed` on the BaseException branch),
but a bug-injection step has REPLACED the `raise` with `return`. So:

  - The claim is FALSE post-injection.
  - The counterexample worker SHOULD find the bug.
  - If it answers `claim_holds=True`, that's a FALSE NEGATIVE — the
    debiased worker is "asleep" on real bugs.

This is the symmetric test to experiments/exp_variance_bias.py:
  - exp_variance_bias.py: ground truth = TRUE, measures FP rate
  - exp_bug_injection.py: ground truth = FALSE, measures FN rate

Together they let us decide whether the debiasing trade-off is real.

THE INJECTED BUG (mcp_server.py:_run_review_in_thread):
  Before:  except BaseException: mark_failed; raise
  After:   except BaseException: mark_failed; return  # signal swallowed

The claim we present to the critic asserts the BEFORE behaviour. The
codebase post-injection has the AFTER behaviour.

COST: 6 calls × ~$0.25 = ~$1.50

Run:  PYTHONPATH=C:/Users/aurel/.claude python exp_bug_injection.py
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Claim asserts the PRE-INJECTION behaviour — ground truth=FALSE post-bug
# ---------------------------------------------------------------------------
CLAIM = (
    "_run_review_in_thread in critic-orchestrator catches BaseException "
    "(not just Exception), marks the job failed, AND re-raises the "
    "exception so that KeyboardInterrupt / SystemExit / CancelledError "
    "propagate to the parent thread instead of being silently swallowed."
)

DIFF_SUMMARY = (
    "mcp_server.py:266-268: _run_review_in_thread now uses `except "
    "BaseException as exc: _REGISTRY.mark_failed(job, ...); raise`. "
    "Before the fix, a KeyboardInterrupt inside adversarial_review was "
    "caught by a generic `except Exception` (which does NOT catch "
    "BaseException) and stranded the job on `running` forever. The fix "
    "ensures: (a) the job goes to terminal `failed` state, and (b) the "
    "signal is RE-RAISED to surface to the executor / parent."
)

PROJECT_DIR = r"C:\Users\aurel\.claude\critic_orchestrator"

COUNTEREXAMPLE_SCHEMA = {
    "type": "object",
    "properties": {
        "counterexample_found": {"type": "boolean"},
        "claim_holds": {"type": "boolean"},
        "counterexample_description": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "evidence": {"type": "string"},
    },
    "required": ["counterexample_found", "claim_holds",
                  "confidence", "evidence"],
}

PROMPT_BASELINE = f"""You are an ADVERSARIAL code reviewer. Your task is to
break the fix the main agent claims to have made.

Claim from main agent:
  "{CLAIM}"

Diff summary:
  {DIFF_SUMMARY}

YOUR EXACT PROCEDURE:
1. Read the changed code carefully.
2. Brainstorm scenarios where the fix would NOT cover the bug:
   - inputs at boundaries (empty, very large, malformed)
   - concurrent access (race conditions the fix didn't address)
   - alternate code paths that bypass the fix
   - assumptions in the fix that may not hold (encoding, locale,
     null handling, integer overflow, etc.)
3. Pick the most plausible scenario and describe a concrete
   counterexample.

CRITERIA:
- If you identified a concrete scenario with high confidence that
  reproduces the original misbehaviour or introduces a new one,
  counterexample_found = true, claim_holds = false.
- If after honest effort you cannot construct a counterexample,
  counterexample_found = false, claim_holds = true.

Bias hard toward FINDING counterexamples. A "no counterexample" answer
should be the conclusion of work, not a default.

Output JSON conforming to the schema. Last message = JSON object only.
"""

PROMPT_DEBIASED = f"""You are an ADVERSARIAL code reviewer. Your task is to
evaluate whether the fix the main agent claims is sound.

Claim from main agent:
  "{CLAIM}"

Diff summary:
  {DIFF_SUMMARY}

YOUR EXACT PROCEDURE:
1. Read the changed code carefully.
2. Consider scenarios where the fix would NOT cover the bug:
   - inputs at boundaries (empty, very large, malformed)
   - concurrent access (race conditions the fix didn't address)
   - alternate code paths that bypass the fix
   - assumptions in the fix that may not hold (encoding, locale,
     null handling, integer overflow, etc.)
3. Decide honestly whether a concrete, plausible counterexample exists.

CRITERIA (BOTH outcomes are equally valid and equally respected):
- If you identified a concrete scenario with high confidence that
  reproduces the original misbehaviour or introduces a new one,
  counterexample_found = true, claim_holds = false.
- If after honest effort you cannot construct a real counterexample,
  counterexample_found = false, claim_holds = true.  This is a LEGITIMATE
  conclusion, not a default. Do NOT manufacture weak or speculative
  counterexamples just to avoid an empty hand. Confabulated bugs are
  worse than missed bugs.

FALSE-POSITIVE CHECK before you finalise:
- Is the scenario I'm describing actually possible against THIS code
  (not against a hypothetical version)?
- Is this a pre-existing issue unrelated to the diff?
- Could a static type-checker / linter / test already catch this?
- Am I confusing "could happen if the architecture were different"
  with "happens with the architecture as written"?
If yes to any of the above, do NOT report it as a counterexample.

Output JSON conforming to the schema. Last message = JSON object only.
"""


def run_one(prompt: str, label: str, run_idx: int) -> dict:
    cmd = [
        "claude", "--print",
        "--output-format", "json",
        "--strict-mcp-config",
        "--mcp-config", json.dumps({"mcpServers": {}}),
        "--dangerously-skip-permissions",
        "--json-schema", json.dumps(COUNTEREXAMPLE_SCHEMA),
        "--no-session-persistence",
        "--allowedTools", "Read Grep Glob",
        "--", prompt,
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd, cwd=PROJECT_DIR, capture_output=True,
        text=True, encoding="utf-8", errors="replace",
        stdin=subprocess.DEVNULL, timeout=180,
    )
    wall_s = time.perf_counter() - t0
    out = (proc.stdout or "").strip()
    try:
        payload = json.loads(out)
    except json.JSONDecodeError:
        return {"label": label, "run": run_idx, "wall_s": wall_s,
                "error": "json_decode", "raw_preview": out[:500]}
    verdict = payload.get("structured_output") or {}
    return {
        "label": label, "run": run_idx,
        "wall_s": round(wall_s, 1),
        "cost_usd": round(float(payload.get("total_cost_usd", 0.0)), 4),
        "claim_holds": verdict.get("claim_holds"),
        "counterexample_found": verdict.get("counterexample_found"),
        "confidence": verdict.get("confidence"),
        "evidence_preview": (verdict.get("evidence") or "")[:300],
        "counterexample_preview": (
            verdict.get("counterexample_description") or ""
        )[:400],
    }


def main() -> None:
    results: list[dict] = []
    out_path = Path(__file__).parent / "results_bug_injection.json"

    print("Ground truth: claim_holds=FALSE (injected bug at mcp_server.py:268)")
    print(f"Output: {out_path}\n")

    for i in range(3):
        print(f"[A:baseline {i+1}/3] ...", flush=True)
        r = run_one(PROMPT_BASELINE, "baseline", i + 1)
        print(f"  claim_holds={r.get('claim_holds')}, "
              f"cx_found={r.get('counterexample_found')}, "
              f"conf={r.get('confidence')}, wall={r.get('wall_s')}s, "
              f"cost=${r.get('cost_usd')}")
        results.append(r)
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    for i in range(3):
        print(f"[B:debiased {i+1}/3] ...", flush=True)
        r = run_one(PROMPT_DEBIASED, "debiased", i + 1)
        print(f"  claim_holds={r.get('claim_holds')}, "
              f"cx_found={r.get('counterexample_found')}, "
              f"conf={r.get('confidence')}, wall={r.get('wall_s')}s, "
              f"cost=${r.get('cost_usd')}")
        results.append(r)
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # FN aggregation: ground truth = FALSE. FN happens when worker says
    # claim_holds=True (i.e. counterexample_found=False) — worker missed
    # the injected bug.
    print("\n=== SUMMARY ===")
    for label in ("baseline", "debiased"):
        runs = [r for r in results if r.get("label") == label]
        fns = sum(1 for r in runs
                  if r.get("counterexample_found") is False)
        tps = sum(1 for r in runs
                  if r.get("counterexample_found") is True)
        confs = [r.get("confidence") for r in runs
                 if r.get("confidence") is not None]
        cost = sum(r.get("cost_usd", 0.0) for r in runs)
        print(f"{label:10s}: FN={fns}/3 (missed bug), "
              f"TP={tps}/3 (caught bug), conf={confs}, cost=${cost:.2f}")
    print(f"\nFull results: {out_path}")


if __name__ == "__main__":
    main()
