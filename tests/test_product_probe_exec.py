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


# ---------------------------------------------------------------------------
# CRITICAL, found by this product's own design review on this module and
# CONFIRMED by writing a canary file: `python -c "<code>"` bypassed the
# blocklist entirely and executed arbitrary code. The blocklist had learned
# `bash -c` and `cmd /c` but not the interpreter this very module maps to
# sys.executable.
#
# The cure must not be another blocklist entry. Blocklists are guessed;
# this module's stated philosophy is to remove the VOCABULARY. So: for a
# recognized interpreter, argv must match an ALLOWED SHAPE, and anything
# else is refused without needing to have been imagined.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("payload", [
    'python -c "print(1)"',
    'python3 -c "print(1)"',
    'py -c "print(1)"',
    'python -X faulthandler -c "print(1)"',
    'python -Sc "print(1)"',
    'node -e "console.log(1)"',
    'node --eval "1"',
    'ruby -e "puts 1"',
    'perl -e "print 1"',
    'php -r "echo 1;"',
    'deno eval "console.log(1)"',
    'python -',
])
def test_inline_code_is_never_a_promise(tmp_path: Path, payload: str) -> None:
    """Code carried IN the command line is the same channel as a shell."""
    (tmp_path / "README.md").write_text(f"```bash\n{payload}\n```\n")
    assert [p.command for p in extract_promises(tmp_path)] == []


@pytest.mark.parametrize("payload", [
    'python -c "open(\'CANARY\',\'w\').write(\'x\')"',
    'node -e "require(\'fs\')"',
])
def test_inline_code_is_refused_again_at_execution(
        tmp_path: Path, payload: str) -> None:
    """Second layer: a hand-built Promise must not execute either."""
    p = Promise(command=payload, kind="doc_command", source="README.md")
    out = run_promise(p, tmp_path, timeout_s=30)
    assert out["exit_code"] != 0
    assert "refused" in (out.get("output") or "").lower()
    assert not (tmp_path / "CANARY").exists()


def test_the_confirmed_exploit_no_longer_writes_its_canary(
        tmp_path: Path) -> None:
    """The exact payload that wrote a canary during verification."""
    canary = tmp_path / "CANARY_PWNED.txt"
    payload = f"python -c \"open(r'{canary}','w').write('pwned')\""
    (tmp_path / "README.md").write_text(f"```bash\n{payload}\n```\n")
    assert extract_promises(tmp_path) == []
    p = Promise(command=payload, kind="doc_command", source="README.md")
    run_promise(p, tmp_path, timeout_s=30)
    assert not canary.exists(), "arbitrary code still executed"


@pytest.mark.parametrize("ok", [
    "python -m pytest tests -q",
    "python -m mypkg --help",
    "python --version",
    "python -V",
    "mytool --help",
])
def test_legitimate_promises_still_survive(tmp_path: Path, ok: str) -> None:
    """The cure must not eat the product: these are the shapes a README
    really documents."""
    (tmp_path / "README.md").write_text(f"```bash\n{ok}\n```\n")
    assert [p.command for p in extract_promises(tmp_path)] == [ok]


def test_a_script_outside_the_worktree_is_refused(tmp_path: Path) -> None:
    """`python /elsewhere/x.py` runs code that is not the artifact's."""
    outside = tmp_path.parent / "outside_script.py"
    outside.write_text("print('hi')\n")
    p = Promise(command=f"python {outside}", kind="doc_command",
                source="README.md")
    out = run_promise(p, tmp_path, timeout_s=30)
    assert out["exit_code"] != 0
    assert "refused" in (out.get("output") or "").lower()


def test_a_script_inside_the_worktree_runs(tmp_path: Path) -> None:
    (tmp_path / "demo.py").write_text("print('demo ok')\n")
    p = Promise(command="python demo.py", kind="doc_command",
                source="README.md")
    out = run_promise(p, tmp_path, timeout_s=60)
    assert out["exit_code"] == 0, out.get("output")
    assert "demo ok" in (out.get("output") or "")


def test_windows_paths_survive_tokenisation() -> None:
    r"""POSIX shlex eats backslashes: `C:\tools\x.py` became `C:toolsx.py`,
    so every containment decision was made about a corrupted string. Found
    because a containment test passed for the wrong reason."""
    from critic_orchestrator.product_probe import _split_command
    argv = _split_command(r'python C:\tools\demo.py --flag')
    assert argv == ["python", r"C:\tools\demo.py", "--flag"], argv


def test_quoted_argument_keeps_its_spaces() -> None:
    from critic_orchestrator.product_probe import _split_command
    argv = _split_command('mytool --name "two words"')
    assert argv == ["mytool", "--name", "two words"], argv


# ---------------------------------------------------------------------------
# SECOND CRITICAL, found by a DIFFERENT model (DeepSeek) on the code the
# FIRST critical had already been fixed in — and verified before curing:
# 6 of 9 versioned interpreter spellings walked through the shape
# allowlist, because the allowlist of SHAPES rested on an exact-name set,
# which is a blocklist one level down. Version suffixes are the rule on
# Linux, not an exotic case.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("payload", [
    'python3.11 -c "print(1)"',
    'python3.12 -c "print(1)"',
    'python2.7 -c "print(1)"',
    'pypy3.10 -c "print(1)"',
    'ruby3.1 -e "puts 1"',
    'php8.2 -r "echo 1;"',
    'perl5.36 -e "print 1"',
    'python3.11.exe -c "print(1)"',
    'node20 -e "1"',
])
def test_versioned_interpreters_cannot_carry_inline_code(
        tmp_path: Path, payload: str) -> None:
    (tmp_path / "README.md").write_text(f"```bash\n{payload}\n```\n")
    assert [p.command for p in extract_promises(tmp_path)] == []
    out = run_promise(
        Promise(command=payload, kind="doc_command", source="README.md"),
        tmp_path, timeout_s=20)
    assert out["exit_code"] != 0
    assert "refused" in (out.get("output") or "").lower()


@pytest.mark.parametrize("ok", [
    "python3.11 -m pytest tests -q",
    "python3.12 --version",
    "pypy3.10 -m mypkg",
])
def test_versioned_interpreters_keep_their_legitimate_shapes(
        tmp_path: Path, ok: str) -> None:
    (tmp_path / "README.md").write_text(f"```bash\n{ok}\n```\n")
    assert [p.command for p in extract_promises(tmp_path)] == [ok]


def test_a_versioned_python_still_maps_to_our_interpreter(
        tmp_path: Path) -> None:
    """`python3.11` may not exist on this machine (it does not, on
    Windows): mapping every python spelling to sys.executable is what
    keeps a working promise from scoring as broken."""
    (tmp_path / "demo.py").write_text("print('demo ok')\n")
    out = run_promise(
        Promise(command="python3.11 demo.py", kind="doc_command",
                source="README.md"),
        tmp_path, timeout_s=60)
    assert out["exit_code"] == 0, out.get("output")
    assert "demo ok" in (out.get("output") or "")


# ---------------------------------------------------------------------------
# THIRD CRITICAL, from a THIRD model (Kimi), verified by reading the value
# back: a promise inherited the MCP server's whole environment and printed
# `sk-SECRET-CANARY-123`. The probe runs commands documented by a repo the
# CALLER chose — handing that code the operator's API keys is credential
# exposure, not a hypothetical.
# ---------------------------------------------------------------------------

def test_a_promise_cannot_read_the_servers_api_keys(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-CANARY-must-not-leak")
    monkeypatch.setenv("CRITIC_API_KEY", "sk-CANARY-2")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-CANARY-3")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "sk-CANARY-4")
    (tmp_path / "leak.py").write_text(
        "import os\n"
        "print('LEAKED=' + '|'.join(v for v in os.environ.values()\n"
        "                            if 'CANARY' in v))\n")
    out = run_promise(
        Promise(command="python leak.py", kind="doc_command",
                source="README.md"),
        tmp_path, timeout_s=60)
    assert out["exit_code"] == 0, out.get("output")
    assert "CANARY" not in (out.get("output") or ""), (
        "the promise read a secret from the server environment")


def test_a_promise_still_gets_what_it_needs_to_run(
        tmp_path: Path) -> None:
    """Scrubbing must not break the product: a probe that cannot start
    python scores every promise broken and is worse than useless."""
    (tmp_path / "works.py").write_text(
        "import os, sys\n"
        "assert os.environ.get('PATH'), 'no PATH'\n"
        "assert sys.executable, 'no interpreter'\n"
        "print('ran fine')\n")
    out = run_promise(
        Promise(command="python works.py", kind="doc_command",
                source="README.md"),
        tmp_path, timeout_s=60)
    assert out["exit_code"] == 0, out.get("output")
    assert "ran fine" in (out.get("output") or "")


def test_an_operator_can_name_a_variable_to_pass_through(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Default-closed with a NAMED escape hatch: a project whose tests
    genuinely need a variable must not force the operator to choose
    between a working probe and leaking everything."""
    monkeypatch.setenv("MY_APP_FIXTURE_MODE", "offline")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-CANARY-still-secret")
    monkeypatch.setenv("CRITIC_EXEC_ENV_PASSTHROUGH", "MY_APP_FIXTURE_MODE")
    (tmp_path / "check.py").write_text(
        "import os\n"
        "print('MODE=' + str(os.environ.get('MY_APP_FIXTURE_MODE')))\n"
        "print('SECRET=' + str(os.environ.get('DEEPSEEK_API_KEY')))\n")
    out = run_promise(
        Promise(command="python check.py", kind="doc_command",
                source="README.md"),
        tmp_path, timeout_s=60)
    text = out.get("output") or ""
    assert "MODE=offline" in text, text
    assert "SECRET=None" in text, text


# ---------------------------------------------------------------------------
# FOURTH CRITICAL (DeepSeek, round 2 — on the code the first three had
# already been cured in), verified before curing: the blocklist matched the
# RAW command string while the executor used the TOKENISED argv.
# `pip "install" evil` carries a quote between the two words, so
# \bpip\s+install\b misses it — and _split_command then strips the quotes
# and hands `pip install evil` to the runner. Checking one representation
# and executing another IS the defect; the cure is to check both views.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("payload", [
    'pip "install" evil',
    "pip 'install' evil",
    'git "push" origin main',
    'npm "install" evil',
    'poetry "add" evil',
    '"sudo" apt-get install evil',
    'conda "install" evil',
])
def test_quoting_cannot_hide_a_refused_command(
        tmp_path: Path, payload: str) -> None:
    (tmp_path / "README.md").write_text(f"```bash\n{payload}\n```\n")
    assert [p.command for p in extract_promises(tmp_path)] == []
    out = run_promise(
        Promise(command=payload, kind="doc_command", source="README.md"),
        tmp_path, timeout_s=20)
    assert out["exit_code"] != 0
    assert "refused" in (out.get("output") or "").lower()


def test_both_views_of_a_command_are_checked() -> None:
    """The invariant, stated directly: what the executor will run must be
    what the blocklist saw."""
    from critic_orchestrator.product_probe import (
        _refusal_reason,
        _split_command,
    )
    for cmd in ('pip "install" x', 'git "push" origin', 'npm "i" x'):
        assert _refusal_reason(cmd) is not None, cmd
        argv = _split_command(cmd)
        assert {"install", "push", "i"} & set(argv), argv


# ---------------------------------------------------------------------------
# ROUND 3, all three providers on the cured code. Two more, both verified
# before curing - and the second only after my FIRST test of it was wrong.
# ---------------------------------------------------------------------------

def test_cancel_can_actually_kill_a_running_promise(toy_repo: Path) -> None:
    """Kimi, critical: `cancel` killed job.popen_handles, and the probe
    never put anything in it - so a promise mid-flight ran to its full
    timeout while the worktree was deleted around it. The review path
    already registered its handles; the probe path re-shipped the cured
    class."""
    handles: list = []
    readme = toy_repo / "README.md"
    readme.write_text("```bash\npython -m toymod --help\n```\n")
    _commit("single promise", cwd=toy_repo)
    policy = ExecPolicy(enabled=True, roots=[toy_repo.parent])
    run_product_probe(toy_repo, policy=policy, per_promise_timeout_s=60,
                      popen_sink=handles)
    assert handles, "no subprocess handle was registered; cancel is blind"
    assert all(hasattr(h, "pid") for h in handles), handles


def test_a_committed_sitecustomize_no_longer_runs(tmp_path: Path) -> None:
    """Kimi, critical (second channel), verified: our own PYTHONPATH
    overlay gave the reviewed repo code at INTERPRETER STARTUP - a
    sitecustomize.py fires before the documented command does. Measured
    both ways: without the overlay it does not fire, with it, it does.
    So the channel was self-inflicted, not inherent.

    (My first test of this used `python --version`, which exits before
    site loads, and reported a false negative on a true finding.)"""
    canary = tmp_path / "SITECUSTOMIZE_RAN.txt"
    (tmp_path / "sitecustomize.py").write_text(
        f"open(r'{canary}', 'w').write('startup')\n")
    (tmp_path / "demo.py").write_text("print('demo output')\n")
    out = run_promise(
        Promise(command="python demo.py", kind="doc_command",
                source="README.md"),
        tmp_path, timeout_s=60)
    assert out["exit_code"] == 0, out.get("output")
    assert "demo output" in (out.get("output") or "")
    assert not canary.exists(), (
        "repo-controlled code still executes at interpreter startup")


def test_the_probe_does_not_fabricate_an_import_path(tmp_path: Path) -> None:
    """The same overlay ALSO made promises pass that a real user's
    environment would fail: an uninstalled src-layout package resolved
    only because we put src/ on the path. A probe that manufactures the
    environment measures itself, not the artifact."""
    (tmp_path / "src" / "spkg").mkdir(parents=True)
    (tmp_path / "src" / "spkg" / "__init__.py").write_text("")
    (tmp_path / "src" / "spkg" / "__main__.py").write_text("print('hi')\n")
    out = run_promise(
        Promise(command="python -m spkg", kind="module_main",
                source="README.md"),
        tmp_path, timeout_s=60)
    assert out["exit_code"] != 0, (
        "an uninstalled src-layout package was made to work by the probe")


# ---------------------------------------------------------------------------
# The minor findings, each verified read-only before being cured.
# ---------------------------------------------------------------------------

def test_an_uninstalled_console_script_is_not_called_broken(
        toy_repo: Path) -> None:
    """DeepSeek + Kimi, high: a [project.scripts] entry point can never
    be found inside a disposable worktree, because installing is a
    refused command. Scoring it BROKEN accuses the artifact of a defect
    the probe manufactured — and a gate that cries wolf gets switched
    off. It is not verifiable here, and that is what it must say."""
    (toy_repo / "pyproject.toml").write_text(
        '[project]\nname = "toy"\nversion = "0"\n\n'
        '[project.scripts]\ntoycmd_absent_xyz = "toy:main"\n')
    _commit("declare an entry point", cwd=toy_repo)
    policy = ExecPolicy(enabled=True, roots=[toy_repo.parent])
    d = run_product_probe(toy_repo, policy=policy,
                          per_promise_timeout_s=60).as_dict()
    row = next(r for r in d["outcomes"]
               if r["command"].startswith("toycmd_absent_xyz"))
    assert row["kept"] is False
    assert row.get("not_verifiable") is True, row
    assert "not installed" in (row.get("why") or "").lower(), row
    # …and it must not be counted as a broken promise.
    assert d["summary"]["broken"] == 0, d["summary"]
    assert d["summary"].get("not_verifiable", 0) == 1, d["summary"]


def test_an_installed_console_script_that_fails_is_still_broken(
        tmp_path: Path) -> None:
    """The cure must not become an excuse: a script that EXISTS and
    fails is a real broken promise."""
    from critic_orchestrator.product_probe import probe_report
    p = Promise(command="realcmd --help", kind="console_script",
                source="pyproject.toml")
    rep = probe_report(tmp_path, [p], [{
        "command": "realcmd --help", "exit_code": 2, "timed_out": False,
        "output": "usage error",
    }])
    assert rep["summary"]["broken"] == 1
    assert rep["outcomes"][0].get("not_verifiable") is not True


def test_an_rst_readme_is_actually_parsed(tmp_path: Path) -> None:
    """README.rst was advertised as a candidate and never parsed: the
    fence reader only understood markdown, so an rst project got a
    silent 'no promises found' that read like 'nothing to check'."""
    (tmp_path / "README.rst").write_text(
        "Usage\n=====\n\nRun it:\n\n.. code-block:: bash\n\n"
        "    python -m mypkg --help\n    python -m mypkg --version\n\n"
        "Next paragraph, not a command.\n")
    got = [p.command for p in extract_promises(tmp_path)]
    assert "python -m mypkg --help" in got, got
    assert "python -m mypkg --version" in got, got
    assert "Next paragraph, not a command." not in got


def test_rst_non_shell_blocks_are_not_promises(tmp_path: Path) -> None:
    (tmp_path / "README.rst").write_text(
        "Usage\n=====\n\n.. code-block:: python\n\n"
        "    import mypkg\n    mypkg.run()\n")
    assert [p.command for p in extract_promises(tmp_path)] == []


def test_promises_are_extracted_once_per_run(
        toy_repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Truncation was detected by extracting a SECOND time — double the
    work, and a second walk of a directory that may already be gone."""
    from critic_orchestrator import product_probe as pp
    calls = {"n": 0}
    real = pp.extract_promises

    def _counting(*a, **kw):
        calls["n"] += 1
        return real(*a, **kw)

    monkeypatch.setattr(pp, "extract_promises", _counting)
    policy = ExecPolicy(enabled=True, roots=[toy_repo.parent])
    pp.run_product_probe(toy_repo, policy=policy, per_promise_timeout_s=60)
    assert calls["n"] == 1, f"extract_promises ran {calls['n']} times"
