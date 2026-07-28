"""Dogfooding: the agentic backend, exercised the way production uses it.

The three smoke runs proved one lens, called directly, on a small file.
That is not evidence that the product works. This exercises the actual
production path and the cases the smoke never touched:

  A. THE MCP TOOL, not the backend object: start_design_review with
     CRITIC_BACKEND=agentic_api, polled to completion — the path an agent
     really takes, including job registry, aggregation and status.
  B. ALL THREE LENSES together, so cross-lens dedupe and corroboration
     see real data instead of fixtures.
  C. A BIG FILE (thousands of lines), where the per-read cap, the read
     budget and the step budget actually bite.
  D. LIVE CANCELLATION: the critical defect found by DeepSeek was fixed
     against MOCKED tests only. Here a real in-flight agentic review is
     cancelled and we check that requests genuinely stop.
  E. CONCURRENCY: two reviews at once through the shared executor.
  F. A CROSS-MODEL COMPARISON on one identical target, so "the agentic
     backend finds real things" becomes a claim with a denominator.

Nothing is written to the reviewed repository: the lenses are read-only.

  python -m critic_orchestrator.experiments.exp_dogfood_agentic [phase...]
      phases: mcp, big, cancel, concurrent, compare   (default: all)
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

from ..smoke_agentic import _load_keys, _repo_root

RESULTS = Path(os.environ.get("TEMP", "/tmp")) / "critic_dogfood"

#: (label, key env, base url, model)
PROVIDERS = [
    ("deepseek", "DEEPSEEK_API_KEY", "https://api.deepseek.com/v1",
     "deepseek-chat"),
    ("glm", "ZAI_API_KEY", "https://api.z.ai/api/paas/v4", "glm-4.6"),
]

SMALL = "critic_orchestrator/job_registry.py"
BIG = "critic_orchestrator/agentic_api.py"


def _configure(provider: tuple[str, str, str, str], keys: dict) -> bool:
    """Point the env at one provider. False when its key is absent."""
    label, key_env, base, model = provider
    key = keys.get(key_env) or os.environ.get(key_env) or ""
    if not key:
        print(f"[{label}] SKIP — no {key_env}", flush=True)
        return False
    os.environ["CRITIC_BACKEND"] = "agentic_api"
    os.environ["CRITIC_API_KEY"] = key
    os.environ["CRITIC_BASE_URL"] = base
    os.environ["CRITIC_MODEL"] = model
    os.environ["CRITIC_MAX_STEPS"] = os.environ.get("CRITIC_MAX_STEPS", "20")
    return True


def _dump(name: str, payload: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    p = RESULTS / f"{name}.json"
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"    -> {p}", flush=True)


def _call(tool: str, args: dict) -> dict:
    from critic_orchestrator import mcp_server
    out = asyncio.run(mcp_server._call_tool_impl(tool, args))
    return json.loads(out[0].text)


def _poll(job_id: str, budget_s: float, label: str) -> dict:
    deadline = time.time() + budget_s
    last = {}
    while time.time() < deadline:
        last = _call("poll_adversarial_review", {"job_id": job_id})
        if last.get("status") != "running":
            return last
        time.sleep(5)
    print(f"[{label}] poll budget expired ({budget_s}s)", flush=True)
    return last


def _summarise(label: str, result: dict) -> None:
    print(f"[{label}] status={result.get('status')} "
          f"lenses_ok={result.get('lenses_ok')} "
          f"failed={result.get('lenses_failed')}", flush=True)
    print(f"[{label}] severities={result.get('by_severity')} "
          f"findings={len(result.get('findings') or [])} "
          f"wall={round((result.get('wall_duration_ms') or 0)/1000, 1)}s",
          flush=True)
    for f in (result.get("findings") or [])[:6]:
        corr = f.get("corroborated_by") or []
        print(f"[{label}]   [{f.get('severity')}] {f.get('title')}"
              + (f"  (+{len(corr)} lens)" if corr else ""), flush=True)
    for w in result.get("workers") or []:
        if not w.get("ok"):
            print(f"[{label}]   !! {w.get('name')} FAILED: "
                  f"{str(w.get('error'))[:160]}", flush=True)


def phase_mcp(root: Path, keys: dict) -> None:
    """A + B: the real MCP path, all three lenses, small module."""
    print("\n=== PHASE mcp: start_design_review, 3 lenses, via MCP ===",
          flush=True)
    for prov in PROVIDERS:
        if not _configure(prov, keys):
            continue
        label = prov[0]
        started = _call("start_design_review", {
            "module_paths": [SMALL], "project_dir": str(root),
            "timeout_s": 600,
        })
        if "error" in started:
            print(f"[{label}] start ERROR: {started['error']}", flush=True)
            continue
        print(f"[{label}] job {started['job_id']} "
              f"n_workers={started['n_workers']}", flush=True)
        polled = _poll(started["job_id"], 900, label)
        result = polled.get("result") or {}
        if result:
            _summarise(label, result)
            _dump(f"mcp_{label}", result)
        else:
            print(f"[{label}] no result: {polled}", flush=True)


def phase_big(root: Path, keys: dict) -> None:
    """C: a big module — do the caps and budgets hold?"""
    print("\n=== PHASE big: the caps and budgets under real pressure ===",
          flush=True)
    lines = len((root / BIG).read_text(encoding="utf-8").splitlines())
    print(f"target {BIG} ({lines} lines)", flush=True)
    for prov in PROVIDERS[:1]:
        if not _configure(prov, keys):
            continue
        label = prov[0]
        started = _call("start_design_review", {
            "module_paths": [BIG], "project_dir": str(root),
            "timeout_s": 600,
        })
        if "error" in started:
            print(f"[{label}] start ERROR: {started['error']}", flush=True)
            continue
        polled = _poll(started["job_id"], 900, label)
        result = polled.get("result") or {}
        if result:
            _summarise(label, result)
            _dump(f"big_{label}", result)


def phase_cancel(root: Path, keys: dict) -> None:
    """D: cancellation, LIVE. The fix was verified on mocks only."""
    print("\n=== PHASE cancel: does an in-flight agentic review stop? ===",
          flush=True)
    if not _configure(PROVIDERS[0], keys):
        return
    started = _call("start_design_review", {
        "module_paths": [BIG], "project_dir": str(root), "timeout_s": 600,
    })
    if "error" in started:
        print(f"start ERROR: {started['error']}", flush=True)
        return
    job_id = started["job_id"]
    print(f"job {job_id} running; letting it work 25s then cancelling",
          flush=True)
    time.sleep(25)
    t0 = time.time()
    cancelled = _call("cancel_adversarial_review", {"job_id": job_id})
    print(f"cancel returned in {time.time()-t0:.2f}s: "
          f"status={cancelled.get('status')} "
          f"killed_workers={cancelled.get('killed_workers')}", flush=True)
    # The real question: do the workers actually stop, and how fast?
    settle_deadline = time.time() + 180
    while time.time() < settle_deadline:
        polled = _call("poll_adversarial_review", {"job_id": job_id})
        if polled.get("status") == "cancelled":
            break
        time.sleep(3)
    polled = _call("poll_adversarial_review", {"job_id": job_id})
    print(f"final status={polled.get('status')} "
          f"elapsed={polled.get('elapsed_s')}s", flush=True)
    _dump("cancel", polled)
    print("NOTE: a real stop shows elapsed close to the cancel moment, "
          "not the full review length.", flush=True)


def phase_concurrent(root: Path, keys: dict) -> None:
    """E: two reviews at once through the shared executor."""
    print("\n=== PHASE concurrent: two agentic reviews at once ===",
          flush=True)
    if not _configure(PROVIDERS[0], keys):
        return
    jobs = []
    for target in (SMALL, BIG):
        started = _call("start_design_review", {
            "module_paths": [target], "project_dir": str(root),
            "timeout_s": 600,
        })
        if "error" in started:
            print(f"start ERROR on {target}: {started['error']}", flush=True)
            continue
        jobs.append((target, started["job_id"]))
        print(f"  {target} -> {started['job_id']}", flush=True)
    for target, job_id in jobs:
        polled = _poll(job_id, 900, target)
        result = polled.get("result") or {}
        print(f"[{target}] status={polled.get('status')} "
              f"findings={len(result.get('findings') or [])}", flush=True)
        if result:
            _dump(f"concurrent_{Path(target).stem}", result)


def phase_compare(root: Path, keys: dict) -> None:
    """F: same target, Claude vs each API model — one denominator."""
    print("\n=== PHASE compare: same target, Claude subscription vs API ===",
          flush=True)
    os.environ.pop("CRITIC_BACKEND", None)
    for k in ("CRITIC_API_KEY", "CRITIC_BASE_URL", "CRITIC_MODEL"):
        os.environ.pop(k, None)
    started = _call("start_design_review", {
        "module_paths": [SMALL], "project_dir": str(root), "timeout_s": 600,
    })
    if "error" in started:
        print(f"claude start ERROR: {started['error']}", flush=True)
        return
    polled = _poll(started["job_id"], 900, "claude")
    result = polled.get("result") or {}
    if result:
        _summarise("claude", result)
        _dump("compare_claude", result)


PHASES = {
    "mcp": phase_mcp, "big": phase_big, "cancel": phase_cancel,
    "concurrent": phase_concurrent, "compare": phase_compare,
}


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    wanted = [a for a in argv if a in PHASES] or list(PHASES)
    root = _repo_root()
    keys = _load_keys()
    print(f"repo root: {root}", flush=True)
    for name in wanted:
        try:
            PHASES[name](root, keys)
        except Exception as exc:
            print(f"[{name}] PHASE CRASHED: {type(exc).__name__}: {exc}",
                  flush=True)
    print("\ndone", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
