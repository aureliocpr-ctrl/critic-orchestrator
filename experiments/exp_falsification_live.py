"""LIVE proof for the falsification wiring: 3 of 3 reviewers on a
third-party provider.

Before this wiring, the agentic backend honestly skipped the
falsification reviewer (needs exec, backend read-only): third-party
providers ran 2 of 3. The claim under test here is the wiring itself,
reviewed the way production reviews it — through the MCP tool, on this
repository, against a REAL commit whose regression test genuinely fails
on the pre-fix baseline.

Success criteria, stated before running:
  * the falsification worker is NOT skipped;
  * its verdict carries test_falsifies_master=true (the probe observed
    FAIL on the baseline and PASS at HEAD);
  * the other two reviewers still run — 3 of 3.

  python -m critic_orchestrator.experiments.exp_falsification_live
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

from ..smoke_agentic import _load_keys, _repo_root

CLAIM = (
    "The falsification probe now out-imports an editable install of the "
    "reviewed package: the worktree is nested under a container named "
    "after the repo and each probe run gets PYTHONPATH=[container, "
    "worktree, worktree/src] first, so the pre-fix run can no longer "
    "silently import post-fix code and report a false 'confirmation "
    "post-hoc'."
)
DIFF = (
    "pinned_exec.py: ephemeral_worktree nests the checkout at "
    "<container>/<repo-name>; run_pinned gains extra_env and a "
    "parent-aware require_worktree check; run_falsification_probe "
    "builds the PYTHONPATH overlay per run."
)
TEST = ("tests/test_falsification_wiring.py::"
        "test_probe_wins_over_an_editable_install_of_the_same_package")
FIXED_FN = "run_falsification_probe"


def _call(tool: str, args: dict) -> dict:
    from critic_orchestrator import mcp_server
    out = asyncio.run(mcp_server._call_tool_impl(tool, args))
    return json.loads(out[0].text)


def main() -> int:
    keys = _load_keys()
    key = keys.get("DEEPSEEK_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        print("no DEEPSEEK_API_KEY — cannot run")
        return 2
    # _repo_root() is the directory CONTAINING the package (right for the
    # read sandbox); execution needs the git repository itself. Passing
    # the parent here produced a live, correctly-worded policy denial
    # ("not a git repository") on the first run of this experiment.
    repo = _repo_root() / "critic_orchestrator"
    os.environ["CRITIC_BACKEND"] = "agentic_api"
    os.environ["CRITIC_API_KEY"] = key
    os.environ["CRITIC_BASE_URL"] = "https://api.deepseek.com/v1"
    os.environ["CRITIC_MODEL"] = "deepseek-chat"
    os.environ["CRITIC_MAX_STEPS"] = "20"
    # The operator opt-in under test.
    os.environ["CRITIC_ALLOW_EXEC"] = "1"
    os.environ["CRITIC_EXEC_ROOTS"] = str(repo)

    t0 = time.time()
    start = _call("start_adversarial_review", {
        "claim": CLAIM,
        "diff_summary": DIFF,
        "test_path": TEST,
        "fixed_function": FIXED_FN,
        "project_dir": str(repo),
        "timeout_s": 600,
    })
    job_id = start.get("job_id")
    print(f"job {job_id} started", flush=True)
    if not job_id:
        print(json.dumps(start, indent=2)[:800])
        return 2

    deadline = time.time() + 1800
    last: dict = {}
    while time.time() < deadline:
        last = _call("poll_adversarial_review", {"job_id": job_id})
        if last.get("status") in ("done", "failed", "cancelled"):
            break
        time.sleep(10)
    dur = time.time() - t0
    print(f"status={last.get('status')} in {dur:.0f}s", flush=True)

    report = last.get("result") or {}
    ran = 0
    fals_ok = False
    for w in report.get("workers", []):
        name = w.get("name")
        err = (w.get("error") or "")
        verdict = w.get("verdict")
        if verdict is not None and not err:
            ran += 1
        line = f"  {name:22s} "
        if err:
            line += f"ERROR/SKIP: {err[:140]}"
        else:
            keysline = {k: verdict.get(k) for k in (
                "test_falsifies_master", "claim_holds",
                "production_caller_exists", "counterexample_found",
                "confidence") if k in (verdict or {})}
            line += json.dumps(keysline)
            if name == "falsification":
                fals_ok = bool(verdict.get("test_falsifies_master"))
        print(line, flush=True)
    print(f"consensus={report.get('consensus')}  workers_ok={ran}/3")
    print(f"FALSIFICATION OBSERVED (not reasoned): {fals_ok}")

    out = Path(os.environ.get("TEMP", "/tmp")) / "critic_falsification_live.json"
    out.write_text(json.dumps(last, indent=2), encoding="utf-8")
    print(f"written {out}")
    return 0 if (ran == 3 and fals_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
