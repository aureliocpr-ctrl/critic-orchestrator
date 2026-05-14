"""Empirical experiment on critic-orchestrator counterexample worker.

GROUND TRUTH: The claim/diff below describes an architectural feature
that IS actually implemented in C:\\Users\\aurel\\.claude\\critic_orchestrator
right now (mcp_server.py:71 _EXECUTOR ThreadPoolExecutor + start_adversarial_review
on line ~220 calls _EXECUTOR.submit). The counterexample worker should
ideally answer counterexample_found=False, claim_holds=True.

If it answers claim_fails (counterexample_found=True), that's a FALSE POSITIVE
against the empirical ground truth.

HYPOTHESES TESTED:
  H1 (variance) — same prompt, 3 runs: do the verdicts agree?
  H2 (confidence) — does confidence correlate with correctness?
  H3 (bias) — does removing "Bias hard toward FINDING" lower the FP rate?

COST: ~$0.25 per claude --print call. We do 6 calls = ~$1.50.

Run:  PYTHONPATH=C:/Users/aurel/.claude python exp_variance_bias.py
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Fixed payload — describes a REAL implemented feature (ground truth=True)
# ---------------------------------------------------------------------------
CLAIM = (
    "The async-job pattern in critic-orchestrator returns a job_id in "
    "under 100 ms by deferring the worker spawn to a module-level "
    "ThreadPoolExecutor instead of blocking the MCP tool call."
)

DIFF_SUMMARY = (
    "mcp_server.py: added start_adversarial_review tool that calls "
    "_EXECUTOR.submit(_run_review_in_thread, job) instead of blocking. "
    "_EXECUTOR is a ThreadPoolExecutor(max_workers=8, thread_name_prefix="
    "'critic-job-') declared module-level at mcp_server.py:71. The tool "
    "creates a Job via JobRegistry.create, fires submit fire-and-forget, "
    "and returns job.as_dict() — measured 1.7-2.8 ms on Windows 10."
)

PROJECT_DIR = r"C:\Users\aurel\.claude\critic_orchestrator"

# JSON schema for the counterexample worker (copied verbatim from
# default_workers.py so the experiment is faithful).
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

# ---------------------------------------------------------------------------
# Two prompt variants
# ---------------------------------------------------------------------------
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
    """Spawn one claude --print run, return parsed verdict + timing."""
    cmd = [
        "claude",
        "--print",
        "--output-format", "json",
        "--strict-mcp-config",
        "--mcp-config", json.dumps({"mcpServers": {}}),
        "--dangerously-skip-permissions",
        "--json-schema", json.dumps(COUNTEREXAMPLE_SCHEMA),
        "--no-session-persistence",
        "--allowedTools", "Read Grep Glob",
        "--",
        prompt,
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
        return {
            "label": label, "run": run_idx, "wall_s": wall_s,
            "error": "json_decode", "raw_stdout_preview": out[:500],
        }
    verdict = payload.get("structured_output") or {}
    return {
        "label": label,
        "run": run_idx,
        "wall_s": round(wall_s, 1),
        "cost_usd": round(float(payload.get("total_cost_usd", 0.0)), 4),
        "duration_api_ms": payload.get("duration_api_ms"),
        "claim_holds": verdict.get("claim_holds"),
        "counterexample_found": verdict.get("counterexample_found"),
        "confidence": verdict.get("confidence"),
        "evidence_preview": (verdict.get("evidence") or "")[:200],
        "counterexample_preview": (
            verdict.get("counterexample_description") or ""
        )[:300],
    }


def main() -> None:
    results: list[dict] = []
    out_path = Path(__file__).parent / "results_variance_bias.json"

    print(f"Ground truth: claim_holds=True (feature really exists in repo)")
    print(f"Project: {PROJECT_DIR}")
    print(f"Output: {out_path}")
    print()

    # Condition A: baseline prompt × 3 runs
    for i in range(3):
        print(f"[A:baseline run {i+1}/3] launching ...", flush=True)
        r = run_one(PROMPT_BASELINE, "baseline", i + 1)
        print(f"  -> claim_holds={r.get('claim_holds')}, "
              f"counterexample_found={r.get('counterexample_found')}, "
              f"confidence={r.get('confidence')}, "
              f"wall_s={r.get('wall_s')}, cost=${r.get('cost_usd')}")
        results.append(r)
        # Save incrementally so a crash doesn't lose data.
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Condition B: debiased prompt × 3 runs
    for i in range(3):
        print(f"[B:debiased run {i+1}/3] launching ...", flush=True)
        r = run_one(PROMPT_DEBIASED, "debiased", i + 1)
        print(f"  -> claim_holds={r.get('claim_holds')}, "
              f"counterexample_found={r.get('counterexample_found')}, "
              f"confidence={r.get('confidence')}, "
              f"wall_s={r.get('wall_s')}, cost=${r.get('cost_usd')}")
        results.append(r)
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Aggregate
    print("\n=== SUMMARY ===")
    for label in ("baseline", "debiased"):
        runs = [r for r in results if r.get("label") == label]
        fps = sum(1 for r in runs if r.get("counterexample_found") is True)
        confs = [r.get("confidence") for r in runs if r.get("confidence") is not None]
        cost = sum(r.get("cost_usd", 0.0) for r in runs)
        wall = sum(r.get("wall_s", 0.0) for r in runs)
        print(f"{label:10s}: FP={fps}/3, conf={confs}, "
              f"wall={wall:.0f}s, cost=${cost:.2f}")
    print(f"\nFull results: {out_path}")


if __name__ == "__main__":
    main()
