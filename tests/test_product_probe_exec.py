"""Tests for EXECUTING product promises — the second built-never-wired.

`extract_promises` existed and `probe_report` scored outcomes, but
nothing produced an outcome: no consumer ran a promise. These tests pin
the executor and its MCP exposure.

Security shape (same trust boundary as pinned_exec, no new sandbox):
  * ExecPolicy first — no execution anywhere the operator did not name;
  * promises run in an ephemeral worktree at HEAD, never in the real
    tree (an uncommitted README line is not a promise yet);
  * argv, no shell — and any command that would REINTRODUCE a shell
    (powershell/cmd/bash -c …) is refused at extraction AND again at
    execution, because two independent layers fail differently;
  * bounded: per-promise timeout with tree-kill, output caps.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from critic_orchestrator.exec_policy import ExecPolicy, ExecPolicyError
from critic_orchestrator.product_probe import (
    Promise,
    extract_promises,
    run_product_probe,
    run_promise,
)


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=str(cwd), capture_output=True,
                   text=True, check=True)


def _commit(msg: str, cwd: Path) -> None:
    _git("add", ".", cwd=cwd)
    _git("-c", "user.email=t@t", "-c", "user.name=t",
         "commit", "-qm", msg, cwd=cwd)


@pytest.fixture()
def toy_repo(tmp_path: Path) -> Path:
    """A committed repo whose README promises one runnable command."""
    root = tmp_path / "toy"
    (root / "toymod").mkdir(parents=True)
    (root / "toymod" / "__init__.py").write_text("")
    (root / "toymod" / "__main__.py").write_text(
        "import sys\nprint('toy help')\nsys.exit(0)\n")
    (root / "README.md").write_text(
        "# Toy\n\nUsage:\n\n```bash\npython -m toymod --help\n```\n")
    _git("init", "-q", cwd=root)
    _commit("initial", cwd=root)
    return root


# ---------------------------------------------------------------------------
# run_promise — one promise, argv, no shell, bounded
# ---------------------------------------------------------------------------

def test_run_promise_help_command_exits_zero(toy_repo: Path) -> None:
    p = Promise(command="python -m toymod --help", kind="doc_command",
                source="README.md")
    out = run_promise(p, toy_repo, timeout_s=60)
    assert out["command"] == p.command
    assert out["exit_code"] == 0, out.get("output")
    assert out["timed_out"] is False
    assert "toy help" in (out.get("output") or "")


def test_run_promise_missing_entry_point_reads_as_127(tmp_path: Path) -> None:
    """A console script that is not installed must score as 'command not
    found' (the report's 127 contract), not crash the probe."""
    p = Promise(command="no_such_cmd_xyz_123 --help",
                kind="console_script", source="pyproject.toml")
    out = run_promise(p, tmp_path, timeout_s=10)
    assert out["exit_code"] == 127
    assert out["timed_out"] is False


def test_run_promise_kills_a_hanging_command(tmp_path: Path) -> None:
    (tmp_path / "sleeper").mkdir()
    (tmp_path / "sleeper" / "__init__.py").write_text("")
    (tmp_path / "sleeper" / "__main__.py").write_text(
        "import time\ntime.sleep(120)\n")
    p = Promise(command="python -m sleeper", kind="module_main",
                source="sleeper/__main__.py")
    out = run_promise(p, tmp_path, timeout_s=3)
    assert out["timed_out"] is True


def test_run_promise_re_refuses_shell_reintroduction(tmp_path: Path) -> None:
    """Defence in depth: even if a refused command reaches execution
    (a bug upstream, a hand-built Promise), it must not run."""
    p = Promise(command="powershell -Command Get-Process",
                kind="doc_command", source="README.md")
    out = run_promise(p, tmp_path, timeout_s=10)
    assert out["exit_code"] != 0
    assert "refused" in (out.get("output") or "").lower()


@pytest.mark.parametrize("evil", [
    "powershell -Command evil",
    "pwsh -c evil",
    "cmd /c dir",
    "bash -c ls",
    "sh -c ls",
    "zsh -c ls",
    "env FOO=1 something",
    "xargs rm",
    "eval something",
])
def test_shell_reintroduction_is_never_extracted(
        tmp_path: Path, evil: str) -> None:
    """The whole point of argv-only execution dies if a documented
    command hands us back a shell. These must not become promises."""
    (tmp_path / "README.md").write_text(
        f"# X\n\n```bash\n{evil}\n```\n")
    got = [p.command for p in extract_promises(tmp_path)]
    assert got == [], got


# ---------------------------------------------------------------------------
# run_product_probe — policy first, worktree always
# ---------------------------------------------------------------------------

def test_probe_end_to_end_keeps_a_real_promise(toy_repo: Path) -> None:
    policy = ExecPolicy(enabled=True, roots=[toy_repo.parent])
    report = run_product_probe(toy_repo, policy=policy,
                               per_promise_timeout_s=60)
    d = report.as_dict()
    assert d["status"] == "promises_kept", json.dumps(d, indent=2)[:800]
    assert d["summary"]["kept"] == 1
    assert d["summary"]["broken"] == 0


def test_probe_refuses_without_policy(toy_repo: Path) -> None:
    with pytest.raises(ExecPolicyError):
        run_product_probe(toy_repo, policy=ExecPolicy(enabled=False))


def test_probe_reads_promises_from_head_not_the_dirty_tree(
        toy_repo: Path) -> None:
    """An uncommitted README edit is not a promise yet: the probe runs
    against HEAD in a worktree, so the real tree's state is irrelevant
    and never touched."""
    readme = toy_repo / "README.md"
    readme.write_text(readme.read_text()
                      + "\n```bash\npython -m uncommitted_thing\n```\n")
    policy = ExecPolicy(enabled=True, roots=[toy_repo.parent])
    report = run_product_probe(toy_repo, policy=policy,
                               per_promise_timeout_s=60)
    d = report.as_dict()
    commands = [r["command"] for r in d["outcomes"]]
    assert "python -m uncommitted_thing" not in commands


def test_probe_cancel_stops_between_promises(toy_repo: Path) -> None:
    """A cancelled probe must stop paying: remaining promises are not
    run and the report says so."""
    # Two promises: add a second doc command, committed.
    readme = toy_repo / "README.md"
    readme.write_text(readme.read_text()
                      + "\n```bash\npython -m toymod --version\n```\n")
    (toy_repo / "toymod" / "__main__.py").write_text(
        "import sys\nprint('toy help')\nsys.exit(0)\n")
    _commit("second promise", cwd=toy_repo)
    policy = ExecPolicy(enabled=True, roots=[toy_repo.parent])
    calls = {"n": 0}

    def _cc() -> bool:
        calls["n"] += 1
        return calls["n"] > 1  # allow the first promise, cancel after

    report = run_product_probe(toy_repo, policy=policy,
                               per_promise_timeout_s=60, cancel_check=_cc)
    d = report.as_dict()
    assert d.get("cancelled") is True
    assert d["summary"].get("not_run", 0) >= 1


# ---------------------------------------------------------------------------
# MCP exposure — the production path
# ---------------------------------------------------------------------------

def _call(tool: str, args: dict) -> dict:
    import asyncio

    from critic_orchestrator import mcp_server
    out = asyncio.run(mcp_server._call_tool_impl(tool, args))
    return json.loads(out[0].text)


def test_mcp_start_product_probe_end_to_end(
        toy_repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import time
    monkeypatch.setenv("CRITIC_ALLOW_EXEC", "1")
    monkeypatch.setenv("CRITIC_EXEC_ROOTS", str(toy_repo.parent))
    start = _call("start_product_probe", {
        "project_dir": str(toy_repo), "per_promise_timeout_s": 60,
    })
    assert start.get("job_id"), start
    deadline = time.time() + 120
    last: dict = {}
    while time.time() < deadline:
        last = _call("poll_adversarial_review", {"job_id": start["job_id"]})
        if last.get("status") in ("done", "failed", "cancelled"):
            break
        time.sleep(0.2)
    assert last.get("status") == "done", json.dumps(last)[:500]
    result = last.get("result") or {}
    assert result.get("kind") == "product_probe"
    assert result.get("status") == "promises_kept"


def test_mcp_probe_fails_fast_without_operator_opt_in(
        toy_repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No CRITIC_ALLOW_EXEC → the caller learns IMMEDIATELY (no job is
    created that would fail minutes later), and the message names the
    knobs."""
    monkeypatch.delenv("CRITIC_ALLOW_EXEC", raising=False)
    monkeypatch.delenv("CRITIC_EXEC_ROOTS", raising=False)
    out = _call("start_product_probe", {"project_dir": str(toy_repo)})
    assert "error" in out
    assert "CRITIC_ALLOW_EXEC" in out["error"]
    assert "job_id" not in out
