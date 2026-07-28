"""Adversarial DESIGN review — the pre-code / no-claim gate.

The post-fix reviewers in `default_workers.py` verify a *claim* against
a *diff*. That contract has a measured failure mode: the proposer writes
the claim, the claim defines the review perimeter (the counterexample
FP-check explicitly discards findings outside it), and the review ends
up scoring the proposer's prudence instead of the artifact. Observed as
four consecutive 3-0 `claim_holds` votes on changes in which
deterministic mutation testing later found five theatre-tests.

This module is the structural cure, not a prompt tweak:

  * The reviewers receive the module paths RAW. There is no `claim`
    parameter — by signature, and by MCP schema. What the code is *for*
    is inferred by the reviewer from the artifact itself; a module whose
    inferred purpose diverges from its real purpose is itself a finding.

  * Three independent lenses, each translated from a human practice
    whose value is empirically documented:

      1. PREMORTEM  — Klein's prospective-hindsight framing ("it HAS
         failed; reconstruct the post-mortem"), which measurably
         outproduces "please critique this". The premise is declared a
         reasoning lens with an honest empty-handed exit, because the
         v0.2.0 experiment on this very repo showed that adversarial
         pressure without an honest exit drives the false-positive
         rate to 100% (see default_workers.py, v0.3.0 note).

      2. PERIMETER  — category-error hunting (murder-board / red-team
         rule: the reviewer may attack the frame, not only the build).
         Tuning errors are out of scope ON PURPOSE: the post-fix
         critics already cover them.

      3. DETECTION  — FMEA-detection / STAMP control-loop audit: for
         every declared safeguard, can it actually fire, and would its
         silent death be noticed? Guards-that-do-not-guard are a
         recurring, measured incident class.

  * A grid of historically frequent failure classes is injected as an
    ADDITIVE search aid ("check these in addition to anything else").
    It must never become a perimeter: the clause "These classes EXTEND
    your search; they never limit it" is part of the contract and is
    pinned by tests.

Aggregation is by findings (deduped, severity-ranked), not by a binary
vote: a design review has no claim to hold or fail.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
import subprocess

from .default_workers import _sanitize_for_prompt, _UNTRUSTED_HEADER
from .orchestrator import (
    CriticReport,
    WorkerSpec,
    WorkerVerdict,
    adversarial_review,
)


# ---------------------------------------------------------------------------
# Historical failure-class grid
# ---------------------------------------------------------------------------
# Each class earned its place by recurring in real incident data (several
# were measured multiple times before being named). The `probe` is the
# operational question a reviewer can act on — not a category label.
DEFAULT_ERROR_GRID: list[dict[str, str]] = [
    {
        "name": "built-never-wired",
        "probe": (
            "Search for capability that exists but is never engaged: "
            "feature flags defaulting to off that no production caller "
            "ever sets, functions/classes with zero non-test callers, "
            "config options never read."
        ),
    },
    {
        "name": "guard-that-does-not-guard",
        "probe": (
            "For every declared safeguard (circuit breaker, rate limit, "
            "rollback, quarantine, validation gate): trace the trigger "
            "path end-to-end. Can it actually fire? A guard whose "
            "threshold, wiring, or reset logic prevents it from ever "
            "tripping is a finding."
        ),
    },
    {
        "name": "green-on-a-slice",
        "probe": (
            "Look for verification that samples a subset (hand-picked "
            "tests, tiny corpus, happy-path fixtures) while the code "
            "implies guarantees about the whole population."
        ),
    },
    {
        "name": "declared-not-measured",
        "probe": (
            "Numbers or guarantees in comments/docs/receipts that no "
            "code path actually measures, and improvement claims with "
            "no harness that could re-measure them."
        ),
    },
    {
        "name": "trusted-input-happy-path",
        "probe": (
            "Input from caller, env, file, or network treated as "
            "well-formed. Trace hostile dataflow: malformed, oversized, "
            "concurrent, adversarial input."
        ),
    },
    {
        "name": "test-theatre",
        "probe": (
            "Tests that would pass even if the feature were broken or "
            "disabled: fixtures that turn the feature off, assertions "
            "that cannot fail, mocks mirroring the implementation."
        ),
    },
    {
        "name": "silent-failopen",
        "probe": (
            "Exception/error paths that degrade to permissive or "
            "default behavior without emitting a receipt, log, or "
            "downgrade marker — the caller cannot tell degraded from "
            "healthy."
        ),
    },
    {
        "name": "state-leak-across-runs",
        "probe": (
            "Module/global state, env vars, singletons, or caches that "
            "leak between runs, tests, or requests; missing reset or "
            "teardown."
        ),
    },
    {
        "name": "time-conflation",
        "probe": (
            "Assertion time vs validity time conflated; stale data "
            "treated as current; TTL or expiry declared but not "
            "enforced."
        ),
    },
    {
        "name": "interface-drift",
        "probe": (
            "Two components that agree in tests but can drift in "
            "production: shape mismatches (batch vs single item, "
            "0-based vs 1-based ids, str vs bytes), silent coercions, "
            "version skew."
        ),
    },
]


_READONLY_TOOLS = ("--allowedTools", "Read Grep Glob")

_SEVERITIES = ("critical", "high", "medium", "low")

#: Cap on the per-lens execution trace carried into the report. Long
#: enough for 20 steps of tool calls with their targets; short enough
#: that one chatty lens cannot bury the findings under its own log.
_MAX_TRACE_CHARS: int = 2_000


def _grid_block(error_grid: list[dict[str, str]] | list[str]) -> str:
    """Render the failure-class grid as an additive search aid.

    The additive clause is contractual (pinned by tests): a grid that
    limited the search would recreate, in softer form, the perimeter
    capture this module exists to remove.
    """
    lines: list[str] = []
    for entry in error_grid:
        if isinstance(entry, dict):
            text = f"{entry.get('name', '?')} — {entry.get('probe', '')}"
        else:
            text = str(entry)
        lines.append(f"  * {_sanitize_for_prompt(text)}")
    return (
        "HISTORICAL FAILURE CLASSES (measured across real incidents in "
        "this development environment). These classes EXTEND your search; "
        "they never limit it — report anything you find whether or not "
        "it matches a class:\n" + "\n".join(lines)
    )


def _targets_block(module_paths: list[str], design_doc: str | None) -> str:
    """Render the review targets as untrusted path data."""
    safe_paths = "\n".join(_sanitize_for_prompt(p) for p in module_paths)
    block = (
        f"{_UNTRUSTED_HEADER}"
        f'<UNTRUSTED_INPUT type="module_paths">\n{safe_paths}\n'
        f"</UNTRUSTED_INPUT>\n"
    )
    if design_doc:
        block += (
            f'\n<UNTRUSTED_INPUT type="design_doc_path">\n'
            f"{_sanitize_for_prompt(design_doc)}\n</UNTRUSTED_INPUT>\n"
        )
    return block


# One output schema for all three reviewers: uniform findings make
# aggregation and cross-worker dedupe trivial, and the `failure_class`
# field lets us measure which grid classes actually produce findings.
_DESIGN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "inferred_purpose": {
            "type": "string",
            "description": (
                "What this module is for, derived from the code alone "
                "(one short paragraph)."
            ),
        },
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "severity": {
                        "type": "string",
                        "enum": list(_SEVERITIES),
                    },
                    "file": {"type": "string"},
                    "line": {"type": "integer"},
                    "failure_class": {
                        "type": "string",
                        "description": (
                            "Matching historical class name, or 'novel'."
                        ),
                    },
                    "mechanism": {
                        "type": "string",
                        "description": (
                            "Concrete causal chain: input/state → wrong "
                            "behavior, anchored in the code as written."
                        ),
                    },
                    "evidence": {
                        "type": "string",
                        "description": (
                            "What in the code supports this (file:line "
                            "quote or paraphrase)."
                        ),
                    },
                },
                "required": ["title", "severity", "file", "mechanism",
                             "evidence"],
            },
        },
        "summary": {"type": "string"},
        "confidence": {
            "type": "number", "minimum": 0.0, "maximum": 1.0,
        },
    },
    "required": ["inferred_purpose", "findings", "summary", "confidence"],
}


_DISCIPLINE_BLOCK = """
DISCIPLINE — applies to every finding, in BOTH directions:
- Every finding must name a concrete mechanism reachable in THIS code as
  written: file:line plus the sequence of events that triggers it.
- Reject a candidate finding if: it describes a hypothetical rewrite
  rather than the code as written; it needs an environmental catastrophe
  unrelated to the code; a type-checker or linter would already catch
  it; it is a style nitpick a senior engineer would not raise.
- Do NOT reject a finding merely because it lies outside what the module
  seems to consider its job: purpose mismatch and missing scope are
  valid findings here.
"""


def _premortem_worker(
    module_paths: list[str],
    design_doc: str | None,
    error_grid: list[dict[str, str]] | list[str],
) -> WorkerSpec:
    prompt = f"""You are conducting a PROJECT PREMORTEM on a software module.

THE PREMISE: six months from now, this module HAS FAILED in production.
The failure was real, costly, and traced back to this code. Do not
debate whether it will fail — reconstruct the most plausible
post-mortem of how it did.

{_targets_block(module_paths, design_doc)}
YOUR PROCEDURE:
1. Read every file listed in module_paths (and the design doc, if
   given). Follow imports inside the same package when needed. Read
   related tests if you find them.
2. Infer what the module is FOR from the artifact alone; write it in
   `inferred_purpose`.
3. Write the post-mortem: the most plausible concrete causal chain that
   produced the failure, then the second most plausible, and so on.
   Anchor every step in file:line evidence. Emit each independent root
   cause as one finding (severity = blast radius of that failure).

{_grid_block(error_grid)}
{_DISCIPLINE_BLOCK}
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
        name="premortem",
        prompt=prompt,
        schema=_DESIGN_SCHEMA,
        extra_args=_READONLY_TOOLS,
        permission_mode="plan",
        requires_execution=True,
    )


def _perimeter_worker(
    module_paths: list[str],
    design_doc: str | None,
) -> WorkerSpec:
    prompt = f"""You are reviewing whether a module solves THE RIGHT PROBLEM —
not whether it is well built. Tuning errors (a wrong threshold, a
missing edge case) are explicitly OUT of your scope; other reviewers
cover those. Your subject is CATEGORY errors: frames that make the
implementation quality irrelevant.

{_targets_block(module_paths, design_doc)}
YOUR PROCEDURE:
1. Read every file listed in module_paths (and the design doc, if
   given). Follow imports inside the same package when needed.
2. Write, in `inferred_purpose`, the problem this module believes it is
   solving — its implicit contract, derived from code and tests alone.
3. Interrogate the frame, and emit each real risk as one finding:
   - PROXY: what real-world property does the core abstraction claim to
     capture — and does it measure that property, or a correlated proxy
     that diverges exactly in the hard cases?
   - LOAD-BEARING ASSUMPTION: which single assumption, if false, makes
     the whole approach irrelevant (not merely mistuned)? Is it checked
     anywhere?
   - CONSUMER MISMATCH: does what callers actually need match what this
     module provides? Look at real call sites.
   - REFRAME: what would this look like if the problem had been framed
     one level up or down? Does that reframe dissolve complexity that
     this design fights?
4. For each finding, `evidence` must state the observable measurement
   that would settle whether the risk is real — an experiment, not an
   opinion. Set `failure_class` to "frame".
{_DISCIPLINE_BLOCK}
If the frame is sound, report zero findings and say so in `summary` —
that conclusion is respected. Manufactured frame-risks are worse than
missed ones.

Output JSON conforming to the schema. The last message you emit must be
ONLY the JSON object; no surrounding prose.
"""
    return WorkerSpec(
        name="perimeter",
        prompt=prompt,
        schema=_DESIGN_SCHEMA,
        extra_args=_READONLY_TOOLS,
        permission_mode="plan",
        requires_execution=True,
    )


def _detection_worker(
    module_paths: list[str],
    design_doc: str | None,
    error_grid: list[dict[str, str]] | list[str],
) -> WorkerSpec:
    prompt = f"""You are auditing a module's CONTROL LOOPS: for every safeguard
or invariant it declares, whether that safeguard can actually fire —
and whether its silent death would be noticed. Failures of detection
outrank failures of function: an invisible breakage of a critical guard
is worse than a visible breakage of a minor one.

{_targets_block(module_paths, design_doc)}
YOUR PROCEDURE:
1. Read every file listed in module_paths. List every declared or
   implied invariant and safeguard: breakers, gates, locks, retries,
   fallbacks, quarantines, receipts, TTLs, thresholds, warnings,
   validation layers.
2. For each one, answer three questions with file:line evidence, and
   emit a finding wherever an answer is "no" or "nothing":
   a. TRIGGER — can it actually fire? Trace the path from the condition
      it guards against to the code that trips it. A breaker nobody
      arms, a threshold never compared, a warning behind a disabled
      flag: findings.
   b. OBSERVER — if this safeguard silently stopped working tomorrow,
      what would notice? A test that would go red? A metric? A receipt
      a caller checks? If the honest answer is "nothing", that is a
      finding even when the safeguard currently works.
   c. RESET — after it fires, what re-arms it? Can it latch forever, or
      re-arm so early it never protects?
3. Severity = (blast radius of the guarded failure) × (invisibility).

{_grid_block(error_grid)}
{_DISCIPLINE_BLOCK}
If every control loop closes, report zero findings and say so in
`summary` — that conclusion is respected.

Output JSON conforming to the schema. The last message you emit must be
ONLY the JSON object; no surrounding prose.
"""
    return WorkerSpec(
        name="detection",
        prompt=prompt,
        schema=_DESIGN_SCHEMA,
        extra_args=_READONLY_TOOLS,
        permission_mode="plan",
        requires_execution=True,
    )


def build_design_workers(
    module_paths: list[str],
    design_doc: str | None = None,
    error_grid: list[dict[str, str]] | list[str] | None = None,
) -> list[WorkerSpec]:
    """Return the three design reviewers, prompted on RAW module paths.

    Deliberately no `claim` / `diff_summary` parameters: the proposer's
    framing must not reach a design reviewer (see module docstring).
    """
    grid = DEFAULT_ERROR_GRID if error_grid is None else error_grid
    return [
        _premortem_worker(module_paths, design_doc, grid),
        _perimeter_worker(module_paths, design_doc),
        _detection_worker(module_paths, design_doc, grid),
    ]


# ---------------------------------------------------------------------------
# Aggregation — findings, not votes
# ---------------------------------------------------------------------------

@dataclass
class DesignReport:
    """Aggregated outcome of a design review.

    Duck-type compatible with what `JobRegistry.mark_done` and the poll
    tool need (an `.as_dict()`), so design jobs flow through the
    existing async job machinery unchanged.
    """

    target: str
    #: "blocking_findings" | "no_blocking_findings" | "incomplete"
    #: | "undecided". ``incomplete`` exists because a review that lost
    #: lenses to timeouts used to report ``no_blocking_findings`` — a
    #: reassuring verdict derived from a third of the review.
    status: str
    findings: list[dict[str, Any]]
    by_severity: dict[str, int]
    workers: list[WorkerVerdict]
    total_cost_usd: float
    wall_duration_ms: int

    def as_dict(self) -> dict[str, Any]:
        per_file: dict[str, int] = {}
        for f in self.findings:
            per_file[f["file"]] = per_file.get(f["file"], 0) + 1
        return {
            "kind": "design_review",
            "target": self.target,
            "status": self.status,
            "lenses_ok": [w.name for w in self.workers if w.ok],
            "lenses_failed": [w.name for w in self.workers if not w.ok],
            "by_severity": dict(self.by_severity),
            "findings_per_file": per_file,
            "findings": list(self.findings),
            "workers": [
                {
                    "name": w.name,
                    "ok": w.ok,
                    "inferred_purpose": (
                        (w.verdict or {}).get("inferred_purpose")
                        if w.ok else None
                    ),
                    "summary": (
                        (w.verdict or {}).get("summary") if w.ok else None
                    ),
                    "confidence": (
                        (w.verdict or {}).get("confidence") if w.ok else None
                    ),
                    "error": w.error,
                    "cost_usd": w.cost_usd,
                    "duration_ms": w.duration_ms,
                    # The backend builds a step-by-step trace precisely so
                    # a lens that failed is explainable. Serialising the
                    # error but dropping the trace made a measured round of
                    # 5 failed lenses diagnosable only by guessing from the
                    # error string. Bounded: a diagnostic, not a payload.
                    "trace": (w.raw_stdout_preview or "")[:_MAX_TRACE_CHARS],
                }
                for w in self.workers
            ],
            "total_cost_usd": self.total_cost_usd,
            "wall_duration_ms": self.wall_duration_ms,
        }


def _normalize_finding(raw: Any, worker: str) -> dict[str, Any] | None:
    """Coerce one raw finding into the aggregate shape; None to drop."""
    if not isinstance(raw, dict):
        return None
    severity = str(raw.get("severity", "")).lower()
    if severity not in _SEVERITIES:
        severity = "medium"
    out: dict[str, Any] = {
        "title": str(raw.get("title", "")).strip() or "(untitled)",
        "severity": severity,
        "file": str(raw.get("file", "")).strip(),
        "failure_class": str(raw.get("failure_class", "novel")) or "novel",
        "mechanism": str(raw.get("mechanism", "")),
        "evidence": str(raw.get("evidence", "")),
        "worker": worker,
        "corroborated_by": [],
        "also_reported_as": [],
    }
    if isinstance(raw.get("line"), int):
        out["line"] = raw["line"]
    return out


def _merge_key(f: dict[str, Any]) -> tuple[str, str, Any]:
    """Identity of a finding for cross-lens dedupe.

    file+line when an anchor line exists — an EXACT signal: on the first
    live run three lenses reported one defect (semantic.py:2139) under
    three different titles and title-dedupe kept all three. Title-based
    fallback only when no line was given. Deliberately no fuzzy
    similarity: near-duplicates on different lines are corroboration to
    keep, and `findings_per_file` surfaces their concentration.
    """
    if isinstance(f.get("line"), int):
        return (f["file"], "L", f["line"])
    return (f["file"], "T", f["title"].lower())


def aggregate_design_report(
    report: CriticReport, *, target: str,
) -> DesignReport:
    """Fold the per-worker verdicts of a design run into a DesignReport.

    Dedupe is deliberately conservative — exact (file, lowercased title)
    only — because two reviewers phrasing one weakness differently is
    signal (independent corroboration), not noise. Corroboration is
    recorded on the surviving finding rather than discarded.
    """
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    order: list[tuple[str, str]] = []
    any_worker_ok = False
    for w in report.workers:
        if not w.ok:
            continue
        any_worker_ok = True
        raw_findings = (w.verdict or {}).get("findings", [])
        if not isinstance(raw_findings, list):
            continue
        for raw in raw_findings:
            norm = _normalize_finding(raw, w.name)
            if norm is None:
                continue
            key = _merge_key(norm)
            if key in merged:
                kept = merged[key]
                if w.name != kept["worker"] and \
                        w.name not in kept["corroborated_by"]:
                    kept["corroborated_by"].append(w.name)
                if norm["title"].lower() != kept["title"].lower() and \
                        norm["title"] not in kept["also_reported_as"]:
                    kept["also_reported_as"].append(norm["title"])
                # The merged finding keeps the most severe assessment.
                sev_rank = {s: i for i, s in enumerate(_SEVERITIES)}
                if sev_rank[norm["severity"]] < sev_rank[kept["severity"]]:
                    kept["severity"] = norm["severity"]
            else:
                merged[key] = norm
                order.append(key)

    sev_rank = {s: i for i, s in enumerate(_SEVERITIES)}
    findings = sorted(
        (merged[k] for k in order),
        key=lambda f: sev_rank[f["severity"]],
    )
    by_severity = {s: 0 for s in _SEVERITIES}
    for f in findings:
        by_severity[f["severity"]] += 1

    lenses_failed = [w for w in report.workers if not w.ok]
    if not any_worker_ok:
        status = "undecided"
    elif by_severity["critical"] or by_severity["high"]:
        # A blocking finding found is found, whatever the other lenses did.
        status = "blocking_findings"
    elif lenses_failed:
        # "Nothing blocking" from a partial review is not nothing blocking.
        status = "incomplete"
    else:
        status = "no_blocking_findings"

    return DesignReport(
        target=target,
        status=status,
        findings=findings,
        by_severity=by_severity,
        workers=list(report.workers),
        total_cost_usd=report.total_cost_usd,
        wall_duration_ms=report.wall_duration_ms,
    )


def design_review(
    module_paths: list[str],
    project_dir: Path,
    *,
    design_doc: str | None = None,
    error_grid: list[dict[str, str]] | list[str] | None = None,
    # 600, not the 300 first shipped: 5 of the 9 design workers measured
    # on 2026-07-27 ran 260-408 s (a 5.7k-line module), so the shipped
    # default would have timed out the majority of the very review that
    # validated the tool. Same class as the daemon-lease constant this
    # repo's own review caught — a number true in its regime, false in
    # the one it must cover.
    timeout: int = 600,
    extra_mcp: dict[str, Any] | None = None,
    max_parallel: int = 3,
    popen_sink: list[subprocess.Popen] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    backend: Any | None = None,
) -> DesignReport:
    """One-call design review: build workers, run, aggregate."""
    workers = build_design_workers(
        module_paths=module_paths,
        design_doc=design_doc,
        error_grid=error_grid,
    )
    target = ", ".join(module_paths)
    report = adversarial_review(
        claim=f"design_review: {target}",  # registry label only — never
        project_dir=project_dir,           # reaches a worker prompt
        workers=workers,
        timeout=timeout,
        extra_mcp=extra_mcp,
        max_parallel=max_parallel,
        popen_sink=popen_sink,
        cancel_check=cancel_check,
        backend=backend,
    )
    return aggregate_design_report(report, target=target)


__all__ = [
    "DEFAULT_ERROR_GRID",
    "DesignReport",
    "aggregate_design_report",
    "build_design_workers",
    "design_review",
]
