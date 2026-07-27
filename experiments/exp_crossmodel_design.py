"""SAME domain, THREE models: DeepSeek + GLM + Kimi on one target.

WHY THIS AND NOT ONE MODEL
One provider gives findings with no denominator. Three independent
models on the identical target give three things a single run cannot:

  * CORROBORATION — a finding two or three models reach independently is
    much harder to dismiss than one model's opinion;
  * SOLO FINDINGS — what only one model saw is where model diversity
    actually pays. Measured precedent in this workspace: DeepSeek found
    a true CRITICAL that four Claude 3-0 votes had missed, and the
    python -c bypass that four of my own review passes walked past;
  * A FALSE-POSITIVE SIGNAL — a claim no other model corroborates is not
    thereby wrong, but it is where verification effort belongs first.

Run as one process per provider, in parallel: the wall time is the
slowest provider, not their sum (Kimi's worst measured case is ~1100 s).

  python -m critic_orchestrator.experiments.exp_crossmodel_design
      [--provider deepseek|glm|kimi]      run ONE provider (child mode)
      [--domain sec|all]                  which module set

Without --provider it launches the children and aggregates their JSON.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from ..smoke_agentic import _load_keys, _repo_root

RESULTS = Path(os.environ.get("TEMP", "/tmp")) / "critic_crossmodel"

#: (label, key env, base url, model, wants streaming)
PROVIDERS: dict[str, tuple[str, str, str, bool]] = {
    "deepseek": ("DEEPSEEK_API_KEY", "https://api.deepseek.com/v1",
                 "deepseek-chat", False),
    "glm": ("ZAI_API_KEY", "https://api.z.ai/api/paas/v4", "glm-4.6", False),
    # Streaming is recommended for reasoning models: measured, it does not
    # change the success rate but compresses the tail (472 s vs 1112 s).
    "kimi": ("MOONSHOT_API_KEY", "https://api.moonshot.ai/v1", "kimi-k3",
             True),
}

#: The security / execution domain — everything that decides WHETHER and
#: WHERE this server runs a command.
DOMAINS: dict[str, list[str]] = {
    "sec": [
        "critic_orchestrator/product_probe.py",
        "critic_orchestrator/exec_policy.py",
        "critic_orchestrator/pinned_exec.py",
    ],
    "all": [
        "critic_orchestrator/product_probe.py",
        "critic_orchestrator/exec_policy.py",
        "critic_orchestrator/pinned_exec.py",
        "critic_orchestrator/agentic_api.py",
    ],
}


def _call(tool: str, args: dict) -> dict:
    from critic_orchestrator import mcp_server
    out = asyncio.run(mcp_server._call_tool_impl(tool, args))
    return json.loads(out[0].text)


def _run_one(provider: str, domain: str) -> int:
    """Child mode: configure one provider and run the design review."""
    key_env, base, model, stream = PROVIDERS[provider]
    keys = _load_keys()
    key = keys.get(key_env) or os.environ.get(key_env) or ""
    if not key:
        print(f"[{provider}] SKIP — no {key_env}", flush=True)
        return 2
    repo = _repo_root()
    os.environ.update({
        "CRITIC_BACKEND": "agentic_api",
        "CRITIC_API_KEY": key,
        "CRITIC_BASE_URL": base,
        "CRITIC_MODEL": model,
        "CRITIC_MAX_STEPS": os.environ.get("CRITIC_MAX_STEPS", "20"),
    })
    if stream:
        os.environ["CRITIC_STREAM"] = "1"
    t0 = time.time()
    start = _call("start_design_review", {
        "module_paths": DOMAINS[domain],
        "project_dir": str(repo),
        "timeout_s": 900,
    })
    job_id = start.get("job_id")
    if not job_id:
        print(f"[{provider}] FAILED to start: {json.dumps(start)[:300]}",
              flush=True)
        return 1
    print(f"[{provider}] job {job_id} running…", flush=True)
    deadline = time.time() + 2400
    last: dict = {}
    while time.time() < deadline:
        last = _call("poll_adversarial_review", {"job_id": job_id})
        if last.get("status") in ("done", "failed", "cancelled"):
            break
        time.sleep(15)
    dur = time.time() - t0
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / f"{provider}_{domain}.json"
    payload = {"provider": provider, "model": model, "domain": domain,
               "duration_s": round(dur, 1), "poll": last}
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    result = last.get("result") or {}
    sev = result.get("by_severity") or {}
    lenses_ok = sum(1 for w in (result.get("lenses") or [])
                    if w.get("ok")) if result.get("lenses") else None
    print(f"[{provider}] {last.get('status')} in {dur:.0f}s — "
          f"status={result.get('status')} by_severity={sev} "
          f"lenses_ok={lenses_ok}", flush=True)
    return 0


def _aggregate(domain: str) -> int:
    """Parent mode: read the children's JSON and compare them."""
    rows: list[dict] = []
    for provider in PROVIDERS:
        p = RESULTS / f"{provider}_{domain}.json"
        if not p.is_file():
            print(f"[{provider}] no result file — did the child run?")
            continue
        rows.append(json.loads(p.read_text(encoding="utf-8")))

    print("\n=== PER PROVIDER ===")
    all_findings: list[tuple[str, dict]] = []
    for row in rows:
        result = (row.get("poll") or {}).get("result") or {}
        sev = result.get("by_severity") or {}
        print(f"{row['provider']:9s} {row['model']:14s} "
              f"{row['duration_s']:7.0f}s  status={result.get('status')}  "
              f"{sev}")
        for f in result.get("findings") or []:
            all_findings.append((row["provider"], f))

    # Corroboration by (file, severity-agnostic title-ish key). Findings
    # are prose, so this is a COARSE grouping by file — deliberately: a
    # semantic match would need a judge, and a judge is another claim.
    by_file: dict[str, dict[str, list[str]]] = {}
    for provider, f in all_findings:
        key = str(f.get("file") or "?")
        entry = by_file.setdefault(key, {})
        entry.setdefault(provider, []).append(
            f"[{f.get('severity')}] {str(f.get('title'))[:90]}")

    print("\n=== BY FILE — who flagged what ===")
    for path in sorted(by_file):
        providers = by_file[path]
        tag = ("CORROBORATED" if len(providers) > 1 else "solo        ")
        print(f"\n{tag}  {path}   ({', '.join(sorted(providers))})")
        for provider in sorted(providers):
            for line in providers[provider]:
                print(f"    {provider:9s} {line}")

    crit = sum(1 for _, f in all_findings
               if f.get("severity") in ("critical", "high"))
    print(f"\ntotal findings: {len(all_findings)}  "
          f"critical+high: {crit}  files touched: {len(by_file)}")
    print(f"artifacts in {RESULTS}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=sorted(PROVIDERS))
    ap.add_argument("--domain", choices=sorted(DOMAINS), default="sec")
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()

    if args.provider:
        return _run_one(args.provider, args.domain)
    if args.aggregate_only:
        return _aggregate(args.domain)

    RESULTS.mkdir(parents=True, exist_ok=True)
    children = []
    for provider in PROVIDERS:
        log = RESULTS / f"{provider}_{args.domain}.log"
        fh = open(log, "w", encoding="utf-8")  # noqa: SIM115
        proc = subprocess.Popen(
            [sys.executable, "-m",
             "critic_orchestrator.experiments.exp_crossmodel_design",
             "--provider", provider, "--domain", args.domain],
            stdout=fh, stderr=subprocess.STDOUT,
            cwd=str(_repo_root()),
        )
        children.append((provider, proc, fh))
        print(f"launched {provider} (pid {proc.pid}) -> {log}", flush=True)

    for provider, proc, fh in children:
        code = proc.wait()
        fh.close()
        print(f"{provider} exited {code}", flush=True)
    return _aggregate(args.domain)


if __name__ == "__main__":
    sys.exit(main())
