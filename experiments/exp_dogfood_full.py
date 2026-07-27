"""FULL dogfooding: every gate this product owns, run on itself.

The pre-publication bar, with the prediction stated first: after the
three wirings, the product reviewed by its own instruments should show
ZERO new built-never-wired residues — the audit's dead-env-flag
detector clean on the new knobs, and the README's promises KEPT by the
probe. Anything else is either a real defect or a false positive to
cure, and both are the point.

  1. run_repo_audit          (deterministic, no LLM)
  2. start_product_probe     (the README as a contract, executed)
  3. start_design_review     (three no-claim lenses, DeepSeek) on the
                             three modules this session touched most

  python -m critic_orchestrator.experiments.exp_dogfood_full
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

from ..smoke_agentic import _load_keys, _repo_root


def _call(tool: str, args: dict) -> dict:
    from critic_orchestrator import mcp_server
    out = asyncio.run(mcp_server._call_tool_impl(tool, args))
    return json.loads(out[0].text)


def _poll(job_id: str, budget_s: float) -> dict:
    deadline = time.time() + budget_s
    last: dict = {}
    while time.time() < deadline:
        last = _call("poll_adversarial_review", {"job_id": job_id})
        if last.get("status") in ("done", "failed", "cancelled"):
            return last
        time.sleep(10)
    return last


def main() -> int:
    repo = _repo_root() / "critic_orchestrator"
    out_dir = Path(os.environ.get("TEMP", "/tmp")) / "critic_dogfood_full"
    out_dir.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []

    # ---- 1. deterministic audit ------------------------------------
    print("== repo audit ==", flush=True)
    audit = _call("run_repo_audit", {"project_dir": str(repo)})
    (out_dir / "audit.json").write_text(json.dumps(audit, indent=2),
                                        encoding="utf-8")
    dead = audit.get("dead_env_flags") or []
    print(f"dead env flags: {len(dead)}")
    for d in dead:
        print(f"  - {d.get('flag')}  ({d.get('file')})")
    devs = audit.get("deviations") or []
    print(f"deviations: {len(devs)} (register, not failures)")
    if dead:
        failures.append(f"audit: {len(dead)} dead env flag(s)")

    # ---- 2. the promises, executed ---------------------------------
    print("\n== product probe (self) ==", flush=True)
    os.environ["CRITIC_ALLOW_EXEC"] = "1"
    os.environ["CRITIC_EXEC_ROOTS"] = str(repo)
    start = _call("start_product_probe", {
        "project_dir": str(repo), "per_promise_timeout_s": 300,
    })
    if not start.get("job_id"):
        print(json.dumps(start)[:400])
        failures.append("probe: did not start")
    else:
        probe = _poll(start["job_id"], budget_s=1800)
        (out_dir / "probe.json").write_text(json.dumps(probe, indent=2),
                                            encoding="utf-8")
        result = probe.get("result") or {}
        print(f"status: {result.get('status')}  "
              f"summary: {result.get('summary')}")
        for row in result.get("outcomes") or []:
            mark = "KEPT " if row.get("kept") else "BROKE"
            why = row.get("why") or ""
            print(f"  [{mark}] {row.get('command')}"
                  + (f"  — {why}" if why else ""))
        if result.get("status") != "promises_kept":
            failures.append(f"probe: {result.get('status')}")

    # ---- 3. design review of the session's modules -----------------
    print("\n== design review (DeepSeek, no-claim lenses) ==", flush=True)
    keys = _load_keys()
    key = keys.get("DEEPSEEK_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        print("no DEEPSEEK_API_KEY — skipping the design review")
        failures.append("design: no key")
    else:
        os.environ["CRITIC_BACKEND"] = "agentic_api"
        os.environ["CRITIC_API_KEY"] = key
        os.environ["CRITIC_BASE_URL"] = "https://api.deepseek.com/v1"
        os.environ["CRITIC_MODEL"] = "deepseek-chat"
        os.environ["CRITIC_MAX_STEPS"] = "20"
        start = _call("start_design_review", {
            "module_paths": [
                "critic_orchestrator/pinned_exec.py",
                "critic_orchestrator/product_probe.py",
                "critic_orchestrator/exec_policy.py",
            ],
            "project_dir": str(repo.parent),
            "timeout_s": 600,
        })
        if not start.get("job_id"):
            print(json.dumps(start)[:400])
            failures.append("design: did not start")
        else:
            design = _poll(start["job_id"], budget_s=1800)
            (out_dir / "design.json").write_text(
                json.dumps(design, indent=2), encoding="utf-8")
            result = design.get("result") or {}
            print(f"status: {result.get('status')}")
            sev = result.get("by_severity") or {}
            print(f"by_severity: {sev}")
            for f in (result.get("findings") or [])[:10]:
                print(f"  [{f.get('severity')}] {f.get('file')}:"
                      f"{f.get('line')}  {str(f.get('title'))[:100]}")
            # A GATE THAT DOES NOT GATE: the first version of this script
            # listed `blocking_findings` among the accepted statuses, so
            # it printed "all self-checks came back clean" underneath a
            # CRITICAL it had just found. That is the exact class this
            # product hunts, in the harness that hunts it.
            blocking = int(sev.get("critical", 0)) + int(sev.get("high", 0))
            if blocking:
                failures.append(
                    f"design: {blocking} blocking finding(s) "
                    f"({sev.get('critical', 0)} critical, "
                    f"{sev.get('high', 0)} high)")
            elif result.get("status") == "undecided":
                failures.append("design: undecided (no lens produced a "
                                "verdict)")

    print(f"\nartifacts in {out_dir}")
    if failures:
        print("GATE RESULT: attention needed ->", "; ".join(failures))
        return 1
    print("GATE RESULT: all self-checks came back clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
