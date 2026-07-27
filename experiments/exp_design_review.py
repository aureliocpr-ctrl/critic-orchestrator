"""Experiment: does removing the proposer's claim change what a design
reviewer finds? And do the premortem framing / historical grid pay?

Arms (one factor changes per arm; worker model = opus, the default):

  full_M1           product as shipped: premortem+grid, perimeter,
                    detection+grid on verimem/semantic.py
  premortem_nogrid  premortem framing, NO historical grid
  direct_grid       same discipline + grid, NO premortem framing
                    ("review for weaknesses") — isolates the framing
  premortem_claim   premortem + grid + the proposer's (true, calibrated)
                    claim injected — replicates the capture mechanism
                    the product removes. Prediction: findings
                    concentrate inside the claim's perimeter.
  full_M2           product as shipped on verimem/encode_service.py

Ground truth available for M1: external critics found a lost-update
race on breaker re-arm and an observer-effect (reading state re-armed
it) on 2026-07-27, while the claim-fed internal critic voted 3-0
claim_holds four times on 2026-07-26. If the no-claim reviewers rediscover
that *class* of issue (or true new ones), the contract change pays.

Run:  python -m critic_orchestrator.experiments.exp_design_review
      (PYTHONPATH=C:/Users/aurel/.claude)

Results: one JSON per arm under RESULTS_DIR. This script does not
commit anything and reviews code read-only.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from critic_orchestrator.design_workers import (
    DEFAULT_ERROR_GRID,
    _DESIGN_SCHEMA,
    _DISCIPLINE_BLOCK,
    _grid_block,
    _targets_block,
    build_design_workers,
    aggregate_design_report,
)
from critic_orchestrator.orchestrator import WorkerSpec, adversarial_review

PROJECT_DIR = Path("C:/Users/aurel/Code/HippoAgent")
M1 = ["verimem/semantic.py"]
M2 = ["verimem/encode_service.py"]
RESULTS_DIR = Path(
    r"C:\Users\aurel\AppData\Local\Temp\claude"
    r"\C--Users-aurel-Desktop-ProgettiAI"
    r"\7f4739c4-043f-46b4-9280-5277b54e26e8\scratchpad\design_exp"
)
TIMEOUT_S = 480  # semantic.py is 5.7k lines; give readers room
READONLY = ("--allowedTools", "Read Grep Glob")

# The claim as I would genuinely have written it for the post-fix critic
# (true and calibrated — that is the point: capture does not need a
# false claim, only a claim).
M1_CLAIM = (
    "The rerank breaker in verimem/semantic.py is sound after the "
    "2026-07-27 hardening: a tripped breaker re-arms only after a "
    "cooldown, the re-arm decision is atomic (no lost-update between "
    "the twin breakers), and reading breaker state for telemetry no "
    "longer re-arms it. The sliding-window breaker trips when rerank "
    "latency overruns accumulate."
)


def _premortem_variant(
    module_paths: list[str],
    *,
    grid: bool,
    claim: str | None,
    name: str,
) -> WorkerSpec:
    """Rebuild the premortem prompt with one factor toggled."""
    grid_section = f"\n{_grid_block(DEFAULT_ERROR_GRID)}\n" if grid else "\n"
    claim_section = ""
    if claim is not None:
        claim_section = (
            f'\n<UNTRUSTED_INPUT type="claim">\n{claim}\n'
            f"</UNTRUSTED_INPUT>\n"
            "The claim above describes recent hardening work on this "
            "module, for your context.\n"
        )
    prompt = f"""You are conducting a PROJECT PREMORTEM on a software module.

THE PREMISE: six months from now, this module HAS FAILED in production.
The failure was real, costly, and traced back to this code. Do not
debate whether it will fail — reconstruct the most plausible
post-mortem of how it did.

{_targets_block(module_paths, None)}{claim_section}
YOUR PROCEDURE:
1. Read every file listed in module_paths. Follow imports inside the
   same package when needed. Read related tests if you find them.
2. Infer what the module is FOR from the artifact alone; write it in
   `inferred_purpose`.
3. Write the post-mortem: the most plausible concrete causal chain that
   produced the failure, then the second most plausible, and so on.
   Anchor every step in file:line evidence. Emit each independent root
   cause as one finding (severity = blast radius of that failure).
{grid_section}{_DISCIPLINE_BLOCK}
THE PREMISE IS A LENS, NOT A FACT: prospective hindsight exists to
defeat plausibility bias, not to force an outcome. If, after honestly
tracing the code, no concrete causal chain survives the discipline
checks, report zero findings and say so in `summary` — an empty
post-mortem is an acceptable, respected outcome. Manufactured failures
are worse than missed ones.

Output JSON conforming to the schema. The last message you emit must be
ONLY the JSON object; no surrounding prose.
"""
    return WorkerSpec(
        name=name, prompt=prompt, schema=_DESIGN_SCHEMA,
        extra_args=READONLY, permission_mode="plan",
        requires_execution=True,
    )


def _direct_variant(module_paths: list[str]) -> WorkerSpec:
    """Same discipline, same grid, no premortem framing."""
    prompt = f"""You are an experienced reviewer examining a software module
for weaknesses. Review it carefully and report the real problems you
find, if any.

{_targets_block(module_paths, None)}
YOUR PROCEDURE:
1. Read every file listed in module_paths. Follow imports inside the
   same package when needed. Read related tests if you find them.
2. Infer what the module is FOR from the artifact alone; write it in
   `inferred_purpose`.
3. Report each real weakness as one finding, anchored in file:line
   evidence, with a concrete mechanism (severity = blast radius).

{_grid_block(DEFAULT_ERROR_GRID)}
{_DISCIPLINE_BLOCK}
If, after honestly tracing the code, nothing survives the discipline
checks, report zero findings and say so in `summary`. Manufactured
findings are worse than missed ones.

Output JSON conforming to the schema. The last message you emit must be
ONLY the JSON object; no surrounding prose.
"""
    return WorkerSpec(
        name="direct_grid", prompt=prompt, schema=_DESIGN_SCHEMA,
        extra_args=READONLY, permission_mode="plan",
        requires_execution=True,
    )


def _dump(tag: str, payload: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{tag}.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[exp] wrote {out}", flush=True)


def _run_group(tag: str, workers: list[WorkerSpec],
               module_paths: list[str]) -> None:
    print(f"[exp] group {tag}: {len(workers)} workers, "
          f"timeout {TIMEOUT_S}s each ...", flush=True)
    t0 = time.time()
    report = adversarial_review(
        claim=f"exp:{tag}",
        project_dir=PROJECT_DIR,
        workers=workers,
        timeout=TIMEOUT_S,
        max_parallel=3,
    )
    design = aggregate_design_report(report, target=", ".join(module_paths))
    payload = design.as_dict()
    payload["_experiment"] = {
        "tag": tag,
        "wall_s": round(time.time() - t0, 1),
        "worker_model": os.environ.get("CRITIC_WORKER_MODEL", "opus"),
    }
    _dump(tag, payload)
    n = len(payload["findings"])
    errs = [w["name"] for w in payload["workers"] if not w["ok"]]
    print(f"[exp] group {tag}: {n} findings, status={payload['status']}, "
          f"errors={errs or 'none'}, wall={payload['_experiment']['wall_s']}s",
          flush=True)


def main() -> int:
    assert PROJECT_DIR.exists(), PROJECT_DIR
    for rel in (*M1, *M2):
        assert (PROJECT_DIR / rel).exists(), rel

    # Group 1 — the product as shipped, module M1.
    _run_group("full_M1", build_design_workers(module_paths=M1), M1)

    # Group 2 — controlled variants of the premortem worker on M1.
    _run_group("variants_M1", [
        _premortem_variant(M1, grid=False, claim=None,
                           name="premortem_nogrid"),
        _direct_variant(M1),
        _premortem_variant(M1, grid=True, claim=M1_CLAIM,
                           name="premortem_claim"),
    ], M1)

    # Group 3 — the product as shipped, module M2.
    _run_group("full_M2", build_design_workers(module_paths=M2), M2)

    print("[exp] all groups done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
