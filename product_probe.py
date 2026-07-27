"""Product probe: run the promises, not just the author's tests.

THE GAP THIS CLOSES
===================
The design lenses READ code. `falsification` runs the AUTHOR'S tests. Both
verify what someone already thought to check. The defects that have cost
most in this workspace are invisible to both:

  * "built, never wired" — counted FOUR separate times in one project;
  * a defect found by probing the built wheel that the entire internal
    suite had missed;
  * "the gate quarantines 75% of the corpus" — true, and 94% of that was
    telemetry, which only surfaced when the product met real data.

The common thread: an author's tests check what the author imagined. The
README checks what the product PROMISED to a user. Nobody runs the README.

So this module treats documentation and packaging metadata as a CONTRACT
and executes it:

  * fenced shell commands in README/docs — what a user is told to type;
  * `[project.scripts]` console entry points — a promise that a command
    exists and starts;
  * `python -m <pkg>` where a `__main__` exists.

Then each promise is either kept or broken, with the real output.

WHAT IS DELIBERATELY NOT A PROMISE
==================================
Extraction is where danger is refused, before any policy question:
`pip install`, `sudo`, `rm`, `curl | sh`, `git push`, `docker run`,
`shutdown` and friends are never executed. A README can legitimately
document a destructive or outbound command, and this probe must not become
the way one gets run. Execution itself reuses `exec_policy` (off by
default, operator-named roots) and the ephemeral worktree from
`pinned_exec` — no second sandbox is invented here.

An entry point's testable promise is `--help`, not its real job with
invented arguments: "the command exists and starts" is checkable and
honest, while guessing arguments would manufacture failures.
"""
from __future__ import annotations

import re
import shlex
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .exec_policy import ExecPolicy
from .pinned_exec import (
    PinnedCommand,
    PinnedExecError,
    ephemeral_worktree,
    run_pinned,
    worktree_pythonpath_env,
)

#: Fences whose content is shell. A ```python fence is not a promise a
#: user types into a terminal.
#:
#: An UNLABELLED fence is deliberately NOT included, and that was measured
#: rather than assumed: on this project's own USAGE.md, bare fences held
#: tree diagrams and pseudo-code, and 9 of the first 10 "promises" the
#: probe extracted were noise. Requiring an explicit shell language loses
#: the occasional real promise and removes almost all of the garbage — a
#: stated trade, not a silent one.
_SHELL_FENCE_LANGS: frozenset[str] = frozenset({
    "bash", "sh", "shell", "console", "zsh", "terminal", "powershell", "ps1",
})

#: Box-drawing and prose markers: a line carrying these is documentation
#: ABOUT commands, not a command.
_NOT_A_COMMAND_RE: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p) for p in (
        r"[├└│─┌┐┘┬┴┼]",          # tree/diagram art
        r"\?\s*$",                  # a question, not an instruction
        r"^\.\.\.",                 # elision
        r"^[A-Z][a-z]+\s+\w+.*:\s*$",  # "See the table below:"
        r"^\w+\([^)]*=",            # pseudo-code call: Agent(x="y")
        r"^cd\s",                   # navigation, not a testable promise
    )
)

#: A plausible command starts with an executable-looking token.
_COMMAND_HEAD_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.\-]*(\s|$)")

#: Never executed, whatever the docs say. Setup, destruction, or anything
#: reaching the network / another machine.
_REFUSED_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE) for p in (
        r"\bpip\s+install\b", r"\bpipx\b", r"\buv\s+(?:pip|add|sync)\b",
        r"\bpoetry\s+(?:install|add)\b", r"\bconda\s+install\b",
        r"\bnpm\s+(?:i|install|publish)\b", r"\byarn\s+add\b",
        r"\bsudo\b", r"\bsu\b\s", r"\bdoas\b",
        r"\brm\b", r"\bdel\b", r"\brmdir\b", r"\bmkfs\b", r"\bdd\b",
        r"\bshutdown\b", r"\breboot\b", r"\bkillall\b",
        r"\bcurl\b", r"\bwget\b", r"\bssh\b", r"\bscp\b", r"\bnc\b",
        r"\bgit\s+(?:push|clone|remote|fetch|pull)\b",
        r"\bdocker\b", r"\bpodman\b", r"\bkubectl\b", r"\bterraform\b",
        r"\btwine\b", r"\bsetx\b", r"\bexport\b", r"\bchmod\b", r"\bchown\b",
        # Shell REINTRODUCTION: this probe's whole defence is argv-only
        # execution, and `powershell -c` / `cmd /c` / `bash -c` hand the
        # attacker a shell back through a single argv element. Interpreter
        # front-doors (env, xargs, eval, start) fall in the same class.
        r"\bpowershell\b", r"\bpwsh\b", r"\bcmd\b", r"\bcommand\b",
        r"\bbash\b", r"\bsh\b", r"\bzsh\b", r"\bfish\b", r"\bksh\b",
        r"\benv\b", r"\bxargs\b", r"\beval\b", r"\bexec\b", r"\bstart\b",
        r"\bformat\b", r"\bfdisk\b", r"\bmount\b",
        r"[|>]", r"&&", r";", r"`", r"\$\(",
    )
)

#: Cap on promises probed in one run. Declared, never silent.
DEFAULT_PROMISE_CAP: int = 25


#: Verbs whose documented promise is "starts and stays up", not "exits 0".
#: Three of the sixteen promises extracted from a real project were
#: servers; scoring those broken on timeout would be a systematic false
#: positive, and a gate that cries wolf gets switched off.
_LONG_RUNNING_RE = re.compile(
    r"\b(?:serve|server|start|daemon|watch|console|repl|listen|tui|"
    r"dev|runserver)\b", re.IGNORECASE,
)


@dataclass(frozen=True)
class Promise:
    """Something the artifact tells a user they can do."""

    command: str
    kind: str          # "doc_command" | "console_script" | "module_main"
    source: str        # where the promise was found

    @property
    def expects_exit(self) -> bool:
        """True when keeping the promise means terminating successfully.

        False for servers and interactive sessions: for those, the promise
        is that the process comes up and stays up, so being killed by the
        probe's grace timeout is SUCCESS, and dying on its own is failure.
        """
        return not _LONG_RUNNING_RE.search(self.command)


def _clean_line(line: str) -> str:
    """Turn a doc line into a command, or "" if it is not one.

    Rejection matters more than extraction here: a probe that runs prose
    produces noise a reader has to filter, and a noisy gate gets ignored.
    """
    s = line.strip()
    for prefix in ("$ ", "> ", "% ", "PS> "):
        if s.startswith(prefix):
            s = s[len(prefix):].strip()
            break
    if s.startswith("#") or s.startswith("//"):
        return ""
    # Trailing explanatory comment: `pytest -q   # 93 passed` — the command
    # is what precedes it. Only when preceded by whitespace, so a `#` that
    # belongs to an argument survives.
    s = re.split(r"\s+#\s", s, maxsplit=1)[0].strip()
    if not s:
        return ""
    if any(p.search(s) for p in _NOT_A_COMMAND_RE):
        return ""
    if not _COMMAND_HEAD_RE.match(s):
        return ""
    return s


def _is_refused(command: str) -> bool:
    return any(p.search(command) for p in _REFUSED_PATTERNS)


def _doc_promises(root: Path) -> list[Promise]:
    out: list[Promise] = []
    candidates: list[Path] = []
    for name in ("README.md", "README.rst", "README.txt", "USAGE.md",
                 "QUICKSTART.md"):
        p = root / name
        if p.is_file():
            candidates.append(p)
    docs = root / "docs"
    if docs.is_dir():
        candidates.extend(sorted(docs.rglob("*.md"))[:20])
    for path in candidates:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel = str(path.relative_to(root)).replace("\\", "/")
        in_fence = False
        fence_is_shell = False
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("```"):
                if in_fence:
                    in_fence, fence_is_shell = False, False
                else:
                    lang = stripped[3:].strip().lower().split()[0] \
                        if stripped[3:].strip() else ""
                    in_fence = True
                    fence_is_shell = lang in _SHELL_FENCE_LANGS
                continue
            if not (in_fence and fence_is_shell):
                continue
            cmd = _clean_line(line)
            if not cmd or _is_refused(cmd):
                continue
            out.append(Promise(command=cmd, kind="doc_command", source=rel))
    return out


def _packaging_promises(root: Path) -> list[Promise]:
    out: list[Promise] = []
    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        try:
            data = tomllib.loads(
                pyproject.read_text(encoding="utf-8", errors="replace"))
        except (OSError, tomllib.TOMLDecodeError):
            data = {}
        scripts = ((data.get("project") or {}).get("scripts") or {})
        for name in sorted(scripts):
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", str(name)):
                continue
            out.append(Promise(
                command=f"{name} --help", kind="console_script",
                source="pyproject.toml",
            ))
    # `python -m pkg` where a __main__ exists.
    for main in sorted(root.glob("*/__main__.py"))[:10]:
        pkg = main.parent.name
        if pkg.startswith(".") or pkg in ("tests", "test", "docs"):
            continue
        out.append(Promise(
            command=f"python -m {pkg} --help", kind="module_main",
            source=str(main.relative_to(root)).replace("\\", "/"),
        ))
    return out


def extract_promises(
    root: Path | str, *, cap: int = DEFAULT_PROMISE_CAP,
) -> list[Promise]:
    """Collect what the artifact promises a user can do.

    Deduplicated by command, sorted for determinism, capped (the caller
    learns about truncation through `probe_report(truncated=...)` — a cap
    that looks like a complete list is the silent-pass this tool exists to
    prevent).
    """
    root = Path(root)
    found = _doc_promises(root) + _packaging_promises(root)
    by_command: dict[str, Promise] = {}
    for p in found:
        if _is_refused(p.command):
            continue
        by_command.setdefault(p.command, p)
    ordered = [by_command[c] for c in sorted(by_command)]
    return ordered[:cap]


def _why_broken(outcome: dict[str, Any]) -> str:
    if outcome.get("timed_out"):
        return "timed out before finishing"
    code = outcome.get("exit_code")
    if code == 127:
        return "command not found — the promised entry point does not exist"
    return f"exited with code {code}"


def probe_report(
    root: Path | str,
    promises: list[Promise],
    outcomes: list[dict[str, Any]],
    *,
    truncated: bool = False,
) -> dict[str, Any]:
    """Fold promises + outcomes into promise-vs-behaviour.

    A promise with NO outcome is reported as `not_run`, never as kept:
    counting an unexecuted promise as satisfied would be exactly the
    silent pass this tool exists to catch.
    """
    root = Path(root)
    by_command = {o.get("command"): o for o in outcomes}
    rows: list[dict[str, Any]] = []
    kept = broken = not_run = 0
    for p in promises:
        o = by_command.get(p.command)
        if o is None:
            not_run += 1
            rows.append({
                "command": p.command, "kind": p.kind, "source": p.source,
                "kept": False, "why": "not run",
            })
            continue
        if p.expects_exit:
            ok = (o.get("exit_code") == 0) and not o.get("timed_out")
            why = "" if ok else _why_broken(o)
        else:
            # A server keeps its promise by still running when the grace
            # period ends; exiting on its own means it fell over.
            ok = bool(o.get("timed_out"))
            why = ("stayed up for the whole grace period (a server is not "
                   "expected to exit)" if ok
                   else f"exited on its own with code {o.get('exit_code')} — "
                        "a documented server should stay up")
        if ok:
            kept += 1
        else:
            broken += 1
        row = {
            "command": p.command, "kind": p.kind, "source": p.source,
            "kept": ok, "exit_code": o.get("exit_code"),
            "timed_out": bool(o.get("timed_out")),
            "expects_exit": p.expects_exit,
            "output": (o.get("output") or "")[:2000],
        }
        if why:
            row["why"] = why
        rows.append(row)

    if not promises:
        status = "no_promises_found"
    elif broken:
        status = "broken_promises"
    elif not_run:
        status = "incomplete"
    else:
        status = "promises_kept"

    report: dict[str, Any] = {
        "kind": "product_probe",
        "target": str(root),
        "status": status,
        "summary": {"kept": kept, "broken": broken, "total": len(promises)},
        "outcomes": rows,
        "promises_truncated": bool(truncated),
    }
    if not_run:
        report["summary"]["not_run"] = not_run
    if not promises:
        report["note"] = (
            "No runnable promise found. Looked for shell fences in "
            "README/USAGE/QUICKSTART/docs, [project.scripts] console entry "
            "points, and python -m targets. A project that documents no "
            "runnable command cannot be probed this way — which is itself "
            "worth knowing."
        )
    return report


# ---------------------------------------------------------------------------
# Execution — the consumer the extraction never had
# ---------------------------------------------------------------------------

#: Grace/timeout default per promise. A --help returns in a second; a
#: documented test command can take minutes, which is what the tool
#: parameter is for.
DEFAULT_PROMISE_TIMEOUT_S: int = 30

#: Output kept per outcome (probe_report clips again for the row).
_MAX_OUTCOME_OUTPUT_CHARS: int = 4_000


def _promise_argv(command: str) -> list[str]:
    """Turn a documented command into argv, or raise PinnedExecError.

    `python` maps to OUR interpreter: the promise is "this works in the
    environment the operator mounted this server in", and a bare
    `python` on PATH may be a different install or absent on Windows.
    """
    argv = shlex.split(command)
    if not argv:
        raise PinnedExecError("empty command")
    if argv[0] in ("python", "python3", "py"):
        argv[0] = sys.executable
    return argv


def run_promise(
    promise: Promise, cwd: Path, *,
    timeout_s: int = DEFAULT_PROMISE_TIMEOUT_S,
    require_worktree: bool = False,
) -> dict[str, Any]:
    """Execute ONE promise in `cwd`. Never raises: every failure mode is
    an outcome row the report can score.

    Re-checks `_is_refused` even though extraction already did: two
    independent layers fail differently, and a hand-built or upstream-
    buggy Promise must not become the day the probe ran `powershell -c`.
    """
    base: dict[str, Any] = {"command": promise.command}
    if _is_refused(promise.command):
        return {**base, "exit_code": -1, "timed_out": False,
                "output": "refused: this command is never executed "
                          "(shell, network, destructive, or setup)"}
    try:
        argv = _promise_argv(promise.command)
    except (PinnedExecError, ValueError) as exc:
        return {**base, "exit_code": -1, "timed_out": False,
                "output": f"refused: unparseable command ({exc})"}
    cmd = PinnedCommand(argv=argv, label=promise.command)
    try:
        res = run_pinned(cmd, cwd, timeout_s=timeout_s,
                         require_worktree=require_worktree,
                         extra_env=worktree_pythonpath_env(Path(cwd)))
    except PinnedExecError as exc:
        # run_pinned raises on spawn failure; "no such executable" is the
        # report's 127 contract (a promised entry point that does not
        # exist), everything else is an honest error row.
        msg = str(exc)
        code = 127 if ("could not start" in msg or "not found" in msg) \
            else -1
        return {**base, "exit_code": code, "timed_out": False,
                "output": msg[:_MAX_OUTCOME_OUTPUT_CHARS]}
    combined = res.stdout or ""
    if res.stderr:
        combined += ("\n[stderr]\n" + res.stderr)
    return {
        **base,
        "exit_code": res.exit_code,
        "timed_out": res.timed_out,
        "duration_s": res.duration_s,
        "output": combined[:_MAX_OUTCOME_OUTPUT_CHARS],
    }


@dataclass
class ProductProbeReport:
    """probe_report's dict behind the .as_dict() the job registry wants."""

    data: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return self.data


def run_product_probe(
    project_dir: Path | str, *,
    policy: ExecPolicy,
    cap: int = DEFAULT_PROMISE_CAP,
    per_promise_timeout_s: int = DEFAULT_PROMISE_TIMEOUT_S,
    cancel_check: Callable[[], bool] | None = None,
) -> ProductProbeReport:
    """Extract the promises at HEAD and run each one in a worktree.

    Policy FIRST (raises ExecPolicyError with the operator knobs named),
    then one ephemeral worktree for the whole run: promises are read
    from the worktree too, so an uncommitted README line is not a
    promise yet and the probe's contract is "what HEAD ships", the same
    commit a user would get.

    `cancel_check` is consulted between promises; a cancelled probe
    reports what it ran, marks the rest not_run, and says `cancelled`.
    """
    repo = policy.check(project_dir)
    with ephemeral_worktree(repo) as wt:
        promises = extract_promises(wt, cap=cap)
        truncated = len(extract_promises(wt, cap=cap + 1)) > len(promises)
        outcomes: list[dict[str, Any]] = []
        cancelled = False
        for p in promises:
            if cancel_check is not None and cancel_check():
                cancelled = True
                break
            outcomes.append(run_promise(p, wt,
                                        timeout_s=per_promise_timeout_s,
                                        require_worktree=True))
    report = probe_report(repo, promises, outcomes, truncated=truncated)
    if cancelled:
        report["cancelled"] = True
    return ProductProbeReport(report)


__all__ = [
    "DEFAULT_PROMISE_CAP",
    "DEFAULT_PROMISE_TIMEOUT_S",
    "ProductProbeReport",
    "Promise",
    "extract_promises",
    "probe_report",
    "run_product_probe",
    "run_promise",
]
