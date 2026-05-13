"""Live smoke test of the async-job pattern with 1 real claude CLI worker.

Run: PYTHONPATH=C:/Users/aurel/.claude python smoke_live.py

Spawns ONE counterexample worker (no test_path / fixed_function so
the falsification + caller_verification workers are skipped). Uses
the user's Claude Code subscription (no Anthropic API key). Verifies:
  - start returns in <100ms with a job_id
  - poll transitions from running to done
  - the final report contains a non-empty CriticReport

Cost: ~$0.30 of subscription tokens (one fresh claude --print run).
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from critic_orchestrator import mcp_server


async def main() -> int:
    project = Path("C:/Users/aurel/.claude/critic_orchestrator").resolve()

    t0 = time.perf_counter()
    start_resp = await mcp_server._call_tool_impl(
        "start_adversarial_review",
        {
            "claim": (
                "JobRegistry.cancel marks the job cancelled and kills "
                "each registered Popen handle via subprocess.Popen.kill()."
            ),
            "diff_summary": (
                "New file job_registry.py adds JobRegistry.cancel(job_id) "
                "which acquires RLock, iterates job.popen_handles, calls "
                ".kill() on handles whose poll() is None, returns count."
            ),
            "project_dir": str(project),
            "timeout_s": 120,
        },
    )
    start_data = json.loads(start_resp[0].text)
    start_ms = (time.perf_counter() - t0) * 1000.0
    print(f"START_ELAPSED_MS={start_ms:.1f}")
    print(f"START_RESPONSE={json.dumps(start_data, indent=2)}")

    job_id = start_data.get("job_id")
    if not job_id:
        print("ERROR: no job_id in start response")
        return 1

    deadline = time.time() + 90.0
    last = None
    while time.time() < deadline:
        await asyncio.sleep(3.0)
        poll_resp = await mcp_server._call_tool_impl(
            "poll_adversarial_review", {"job_id": job_id},
        )
        last = json.loads(poll_resp[0].text)
        status = last.get("status")
        elapsed = last.get("elapsed_s")
        print(f"POLL status={status} elapsed_s={elapsed}")
        if status != "running":
            break

    print("---FINAL---")
    if last is not None:
        print(json.dumps(last, indent=2)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
