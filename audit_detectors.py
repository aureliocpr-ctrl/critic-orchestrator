"""Deterministic audit detectors — the non-LLM half of the audit gate.

Where the design reviewers (design_workers.py) are judgment, this module
is bookkeeping. It targets two failure classes that recurred in real
incident data and are mechanically findable — no model, no network, and
therefore no confabulation:

  * DEAD ENV FLAGS → the "built-never-wired" class (a capability
    shipped, gated behind an env flag defaulting OFF, and nothing —
    no config, doc, script, or assignment — ever turns it on). This
    class was measured four separate times before being named; a grep
    would have caught every instance.

  * DEVIATION REGISTER → the "normalization of deviance" class: a
    declared, accepted limitation ("KNOWN LIMIT", "for now",
    "deliberately") that silently becomes the status quo because
    nothing tracks its age. The register makes age visible so an old
    deroga is a queue item, not wallpaper.

Deliberate design constraints:

  * Findings are CANDIDATES for a human/agent judge — the detectors
    optimize recall on their class and report the evidence needed to
    triage fast (tiers, read sites, references), they do not pretend
    judgment.
  * No silent caps: when output is truncated, the report says so
    (`deviations_truncated`) — a truncated list that looks complete
    reads as "covered everything", which is itself the green-on-a-slice
    failure class.
  * Deterministic output ordering, so two runs diff cleanly.
"""
from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

# Directories that host vendored/generated code — scanning them yields
# findings nobody owns.
_SKIP_DIRS: frozenset[str] = frozenset({
    ".git", ".hg", ".svn", ".venv", "venv", "env", "node_modules",
    "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "dist", "build", "site-packages", ".tox", ".eggs",
})

_CODE_EXT = {".py"}
_DOC_EXT = {".md", ".rst", ".txt"}
_CONFIG_EXT = {".yml", ".yaml", ".toml", ".json", ".ini", ".cfg",
               ".ps1", ".sh", ".bat", ".cmd", ".env"}

# os.environ.get("X"[, default]) / os.getenv("X"[, default])
_ENV_READ_SIMPLE_RE = re.compile(
    r"os\.(?:environ\.get|getenv)\(\s*['\"](?P<flag>[A-Z][A-Z0-9_]{2,})['\"]"
    r"\s*(?:,\s*(?P<default>[^)]*))?\)"
)

_FALSY_DEFAULTS: frozenset[str] = frozenset({
    "'0'", '"0"', "''", '""', "'false'", '"false"', "'off'", '"off"',
    "'no'", '"no"', "none", "false", "0", "0.0", "'0.0'", '"0.0"',
})

# `os.environ.get('X') or <fallback>` — the fallback, not the .get()
# default, decides the effective value. First live run reported
# ENGRAM_MODEL_LOCK_TIMEOUT_S (default `or 90`) as unwired: FP.
_OR_FALLBACK_RE = re.compile(r"^\s*or\s+(?P<fb>[^#\n]+)")

# Markers that declare a deviation. Checked case-sensitively where the
# convention is upper-case, case-insensitively where prose varies.
_DEVIATION_MARKERS_EXACT: tuple[str, ...] = (
    "KNOWN LIMIT", "TODO", "FIXME", "HACK", "XXX",
)
_DEVIATION_MARKERS_CI: tuple[str, ...] = (
    "not fixed here", "deliberately still", "for now", "declared, not a defect",
)


def _iter_files(root: Path, exts: set[str]) -> list[Path]:
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames
            if d not in _SKIP_DIRS and not d.startswith(".")
        ]
        for fn in filenames:
            p = Path(dirpath) / fn
            if p.suffix.lower() in exts or fn.startswith(".env"):
                out.append(p)
    return sorted(out)


def _rel(root: Path, p: Path) -> str:
    return str(p.relative_to(root)).replace("\\", "/")


def _is_test_path(rel: str) -> bool:
    low = rel.lower()
    return "test" in low or "conftest" in low


def find_dead_env_flags(root: Path) -> list[dict[str, Any]]:
    """Find env flags read with a falsy default that nothing ever sets.

    Returns one entry per flag (read sites merged), tiers:
      * "unwired"   — referenced nowhere outside its read sites
      * "test-only" — referenced only by tests
    Flags referenced in docs/config/assignments are wired → not reported.
    """
    root = Path(root)
    reads: dict[str, dict[str, Any]] = {}
    # Pass 1 — collect reads with defaults.
    for p in _iter_files(root, _CODE_EXT):
        rel = _rel(root, p)
        if _is_test_path(rel):
            continue  # a read inside a test is not a product capability
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            for m in _ENV_READ_SIMPLE_RE.finditer(line):
                flag = m.group("flag")
                raw_default = (m.group("default") or "").strip()
                orm = _OR_FALLBACK_RE.match(line[m.end():])
                if orm:
                    fb = orm.group("fb").split(")")[0].strip().lower()
                    if fb not in _FALSY_DEFAULTS:
                        # Effective default is the truthy/unjudgeable
                        # `or` fallback → live by default, drop.
                        reads.pop(flag, None)
                        reads.setdefault("!truthy:" + flag, {})
                        continue
                default_l = raw_default.lower()
                if raw_default and default_l not in _FALSY_DEFAULTS:
                    # Default truthy (or an expression we cannot judge):
                    # live by default — not the built-never-wired class.
                    reads.pop(flag, None)
                    # Remember it is truthy somewhere so a later falsy
                    # read of the same flag does not resurrect it.
                    reads.setdefault("!truthy:" + flag, {})
                    continue
                if "!truthy:" + flag in reads:
                    continue
                entry = reads.setdefault(flag, {
                    "flag": flag,
                    "default": None,
                    "read_sites": [],
                })
                entry["read_sites"].append(f"{rel}:{i}")
                if raw_default:
                    entry["default"] = raw_default

    reads = {k: v for k, v in reads.items() if not k.startswith("!truthy:")}
    if not reads:
        return []

    # Pass 2 — hunt references anywhere else (docs, config, code, tests).
    refs: dict[str, dict[str, list[str]]] = {
        f: {"tests": [], "docs": [], "config": [], "code": []}
        for f in reads
    }
    all_exts = _CODE_EXT | _DOC_EXT | _CONFIG_EXT
    for p in _iter_files(root, all_exts):
        rel = _rel(root, p)
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for flag, entry in reads.items():
            if flag not in text:
                continue
            # `os.environ['X'] = ...` / `os.environ['X']` is a real use the
            # .get()/getenv() regex does not model. Treat any subscript
            # occurrence as a reference so a blind spot never masquerades
            # as evidence of deadness.
            read_lines = {
                int(site.rsplit(":", 1)[1])
                for site in entry["read_sites"]
                if site.rsplit(":", 1)[0] == rel
            }
            for i, line in enumerate(text.splitlines(), start=1):
                if flag not in line:
                    continue
                if i in read_lines:
                    continue  # the read site itself is not a reference
                loc = f"{rel}:{i}"
                if _is_test_path(rel):
                    refs[flag]["tests"].append(loc)
                elif p.suffix.lower() in _DOC_EXT:
                    refs[flag]["docs"].append(loc)
                elif p.suffix.lower() in _CONFIG_EXT or p.name.startswith(".env"):
                    refs[flag]["config"].append(loc)
                else:
                    refs[flag]["code"].append(loc)

    out: list[dict[str, Any]] = []
    for flag in sorted(reads):
        r = refs[flag]
        if r["docs"] or r["config"] or r["code"]:
            continue  # wired
        tier = "test-only" if r["tests"] else "unwired"
        out.append({
            "flag": flag,
            "default": reads[flag]["default"],
            "read_sites": sorted(reads[flag]["read_sites"]),
            "references": {k: sorted(v) for k, v in r.items()},
            "tier": tier,
        })
    return out


def _blame_age_days(root: Path, rel: str, line: int) -> float | None:
    """Age in days of a line per git blame; None when unavailable."""
    try:
        proc = subprocess.run(
            ["git", "blame", "-L", f"{line},{line}", "--porcelain", rel],
            cwd=str(root), capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    for ln in proc.stdout.splitlines():
        if ln.startswith("committer-time "):
            try:
                ts = int(ln.split()[1])
            except (IndexError, ValueError):
                return None
            return max(0.0, (time.time() - ts) / 86400.0)
    return None


def find_deviations(
    root: Path, *, with_age: bool = True, cap: int = 200,
) -> tuple[list[dict[str, Any]], bool]:
    """Collect declared-deviation markers with (optionally) their age.

    Returns ``(entries, truncated)``. Sorted oldest-first when ages are
    known, else file order.

    The truncation flag is RETURNED, never stashed on the function
    object: the first version set ``find_deviations._last_truncated`` and
    ``audit_repo`` read it back, which is process-global mutable state in
    a server whose executor runs 8 jobs concurrently — two audits with
    different caps would overwrite each other's flag, so the mechanism
    that exists to prevent a silent cap could itself go silent. That is
    the state-leak-across-runs class this module's own grid lists.
    """
    root = Path(root)
    found: list[dict[str, Any]] = []
    for p in _iter_files(root, _CODE_EXT | _DOC_EXT):
        rel = _rel(root, p)
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            marker = next(
                (mk for mk in _DEVIATION_MARKERS_EXACT if mk in line),
                None,
            ) or next(
                (mk for mk in _DEVIATION_MARKERS_CI
                 if mk in line.lower()),
                None,
            )
            if marker is None:
                continue
            found.append({
                "file": rel, "line": i, "marker": marker,
                "text": line.strip()[:200],
                "age_days": None,
            })
    truncated = len(found) > cap
    found = found[:cap]
    if with_age:
        for d in found:
            age = _blame_age_days(root, d["file"], d["line"])
            d["age_days"] = round(age, 1) if age is not None else None
        found.sort(key=lambda d: -(d["age_days"] or -1.0))
    return found, truncated


def audit_repo(
    root: Path | str, *, with_age: bool = True, deviations_cap: int = 200,
) -> dict[str, Any]:
    """Run every deterministic detector; deterministic, serializable."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    dead = find_dead_env_flags(root)
    devs, truncated = find_deviations(
        root, with_age=with_age, cap=deviations_cap,
    )
    return {
        "kind": "repo_audit",
        "repo": str(root),
        "dead_flags": dead,
        "deviations": devs,
        "deviations_truncated": truncated,
        "summary": {
            "dead_flags": len(dead),
            "deviations": len(devs),
        },
    }


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json as _json
    import sys as _sys

    ap = argparse.ArgumentParser(
        description="Deterministic repo audit: dead env flags + deviation register.",
    )
    ap.add_argument("repo", help="repository root to audit")
    ap.add_argument("--no-age", action="store_true",
                    help="skip git-blame ages (faster)")
    ap.add_argument("--cap", type=int, default=200,
                    help="max deviation entries (default 200)")
    args = ap.parse_args(argv)
    report = audit_repo(
        Path(args.repo), with_age=not args.no_age,
        deviations_cap=args.cap,
    )
    _json.dump(report, _sys.stdout, indent=2)
    _sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
