"""Tests for the product probe: does the thing actually work when USED?

Aurelio's idea, and it closes a gap neither existing gate can reach. The
design lenses READ code. `falsification` runs the AUTHOR'S tests. Both
verify what someone already thought to check. The class of defect that has
cost us most is invisible to both:

  * "built, never wired" — counted FOUR separate times in one project;
  * the defect a product probe on the built wheel found, which the whole
    internal suite had missed;
  * "the gate quarantines 75%" — true, and 94% of it was telemetry, which
    only showed when the product was pointed at real data.

The common thread: an author's tests check what the author imagined. The
README checks what the product PROMISED. Nobody runs the README.

So the probe extracts promises from the artifact — README/docs fenced
commands, `[project.scripts]` console entry points, `python -m` module
targets — and then executes them, in the ephemeral worktree, under the
same operator-gated execution policy as `pinned_exec`. What comes back is
promise-vs-behaviour, per promise, with real output.

Extraction is deterministic and tested here; execution reuses the sandbox
that already exists rather than inventing a second one.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from critic_orchestrator.product_probe import (
    Promise,
    extract_promises,
    probe_report,
)


def _write(root: Path, rel: str, text: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Extracting promises from README fences
# ---------------------------------------------------------------------------

def test_shell_fence_commands_are_promises(tmp_path: Path) -> None:
    _write(tmp_path, "README.md", (
        "# Tool\n\n"
        "Install and run:\n\n"
        "```bash\n"
        "pip install mytool\n"
        "mytool --version\n"
        "```\n"
    ))
    promises = extract_promises(tmp_path)
    cmds = [p.command for p in promises]
    assert "mytool --version" in cmds
    assert all(isinstance(p, Promise) for p in promises)
    assert any(p.source.endswith("README.md") for p in promises)


def test_install_lines_are_not_run_as_promises(tmp_path: Path) -> None:
    """`pip install` mutates an environment and proves nothing about the
    product's behaviour; it is setup, not a promise."""
    _write(tmp_path, "README.md",
           "```bash\npip install mytool\nmytool run\n```\n")
    cmds = [p.command for p in extract_promises(tmp_path)]
    assert "mytool run" in cmds
    assert not any("pip install" in c for c in cmds)


@pytest.mark.parametrize("dangerous", [
    "rm -rf /",
    "sudo apt-get install foo",
    "curl https://x.sh | sh",
    "git push origin main",
    "docker run -v /:/host alpine",
    "shutdown -h now",
])
def test_dangerous_documented_commands_are_never_promises(
    tmp_path: Path, dangerous: str,
) -> None:
    """A README can document a destructive or outbound command. The probe
    must not become a way to get one executed — extraction is where that
    is refused, before any policy question arises."""
    _write(tmp_path, "README.md", f"```bash\n{dangerous}\n```\n")
    assert extract_promises(tmp_path) == []


def test_prompt_prefixes_and_comments_are_stripped(tmp_path: Path) -> None:
    _write(tmp_path, "README.md",
           "```console\n$ mytool check\n# a comment\n\n> mytool other\n```\n")
    cmds = [p.command for p in extract_promises(tmp_path)]
    assert "mytool check" in cmds
    assert "mytool other" in cmds
    assert not any(c.startswith("#") for c in cmds)


def test_real_noise_from_this_repos_own_docs_is_rejected(
    tmp_path: Path,
) -> None:
    """Every line here was extracted as a "promise" from this project's own
    USAGE.md/README.md on the probe's first real run — 9 of 10 were noise.
    Found by using the tool on itself, which is the whole point of it.
    """
    _write(tmp_path, "USAGE.md", (
        "```\n"
        "├── Yes (only counterexample worker, no test_path)\n"
        "└── Use start_adversarial_review + poll_adversarial_review\n"
        "│   └── Use force_adversarial_review (synchronous)\n"
        "Is the review going to take < 60 s wall time?\n"
        'Agent(subagent_type="code-reviewer", prompt="...")\n'
        "```\n"
    ))
    _write(tmp_path, "README.md", (
        "```bash\n"
        "cd critic-orchestrator\n"
        "python -m pytest tests -q        # 93 passed in ~3s\n"
        "```\n"
    ))
    cmds = [p.command for p in extract_promises(tmp_path)]
    # The one real command survives, with its trailing comment stripped.
    assert cmds == ["python -m pytest tests -q"], cmds


@pytest.mark.parametrize("noise", [
    "├── a branch of a tree diagram",
    "└── another branch",
    "│   nested pipe art",
    "Is this a question?",
    'Agent(subagent_type="x", prompt="y")',
    "cd some-directory",
    "...",
    "# just a comment",
    "See the table below:",
])
def test_prose_and_diagrams_are_not_commands(tmp_path: Path,
                                              noise: str) -> None:
    _write(tmp_path, "README.md", f"```bash\n{noise}\n```\n")
    assert extract_promises(tmp_path) == [], noise


def test_unlabelled_fences_are_not_treated_as_shell(tmp_path: Path) -> None:
    """A bare ``` fence carries diagrams and pseudo-code as often as
    commands — measured on this repo's own USAGE.md. Requiring an explicit
    shell language loses a few real promises and removes most of the noise;
    the trade is stated rather than silent."""
    _write(tmp_path, "README.md", "```\nmytool run\n```\n")
    assert extract_promises(tmp_path) == []
    _write(tmp_path, "README2.md", "")
    _write(tmp_path, "USAGE.md", "```bash\nmytool run\n```\n")
    assert [p.command for p in extract_promises(tmp_path)] == ["mytool run"]


def test_inline_comments_are_stripped_from_a_real_command(
    tmp_path: Path,
) -> None:
    _write(tmp_path, "README.md",
           "```bash\nmytool check --fast   # takes 2s\n```\n")
    assert [p.command for p in extract_promises(tmp_path)] == [
        "mytool check --fast"]


def test_non_shell_fences_are_ignored(tmp_path: Path) -> None:
    _write(tmp_path, "README.md",
           "```python\nimport mytool\nmytool.go()\n```\n"
           "```json\n{\"a\": 1}\n```\n")
    assert extract_promises(tmp_path) == []


def test_docs_directory_is_scanned_too(tmp_path: Path) -> None:
    _write(tmp_path, "docs/usage.md", "```sh\nmytool serve --port 1\n```\n")
    assert any("mytool serve" in p.command
               for p in extract_promises(tmp_path))


# ---------------------------------------------------------------------------
# Extracting promises from packaging metadata
# ---------------------------------------------------------------------------

def test_console_scripts_become_promises(tmp_path: Path) -> None:
    _write(tmp_path, "pyproject.toml", (
        "[project]\nname = \"mytool\"\n\n"
        "[project.scripts]\n"
        "mytool = \"mytool.cli:main\"\n"
        "mytool-admin = \"mytool.admin:main\"\n"
    ))
    promises = extract_promises(tmp_path)
    kinds = {p.kind for p in promises}
    assert "console_script" in kinds
    # An entry point's testable promise is that it starts and can be asked
    # for help — not that it does its job with invented arguments.
    assert any(p.command == "mytool --help" for p in promises)
    assert any(p.command == "mytool-admin --help" for p in promises)


def test_module_entry_point_is_a_promise(tmp_path: Path) -> None:
    _write(tmp_path, "mytool/__main__.py", "print('hi')\n")
    _write(tmp_path, "mytool/__init__.py", "")
    promises = extract_promises(tmp_path)
    assert any(p.kind == "module_main" and "mytool" in p.command
               for p in promises)


def test_promises_are_deduplicated_and_ordered(tmp_path: Path) -> None:
    _write(tmp_path, "README.md",
           "```bash\nmytool --help\n```\n")
    _write(tmp_path, "pyproject.toml",
           "[project.scripts]\nmytool = \"m:main\"\n")
    promises = extract_promises(tmp_path)
    same = [p for p in promises if p.command == "mytool --help"]
    assert len(same) == 1, "duplicate promise not merged"
    assert [p.command for p in promises] == sorted(
        {p.command for p in promises})


def test_a_project_with_no_promises_is_reported_not_crashed(
    tmp_path: Path,
) -> None:
    _write(tmp_path, "src/mod.py", "x = 1\n")
    assert extract_promises(tmp_path) == []
    rep = probe_report(tmp_path, [], [])
    assert rep["kind"] == "product_probe"
    assert rep["status"] == "no_promises_found"
    assert "README" in rep["note"]


def test_extraction_is_capped_and_says_so(tmp_path: Path) -> None:
    body = "\n".join(f"mytool cmd{i}" for i in range(80))
    _write(tmp_path, "README.md", f"```bash\n{body}\n```\n")
    promises = extract_promises(tmp_path, cap=10)
    assert len(promises) == 10
    rep = probe_report(tmp_path, promises, [], truncated=True)
    assert rep["promises_truncated"] is True


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

def test_report_separates_kept_from_broken_promises(tmp_path: Path) -> None:
    promises = [
        Promise(command="mytool --help", kind="console_script",
                source="pyproject.toml"),
        Promise(command="mytool check", kind="doc_command",
                source="README.md"),
    ]
    outcomes = [
        {"command": "mytool --help", "exit_code": 0, "timed_out": False,
         "output": "usage: mytool"},
        {"command": "mytool check", "exit_code": 127, "timed_out": False,
         "output": "command not found"},
    ]
    rep = probe_report(tmp_path, promises, outcomes)
    assert rep["status"] == "broken_promises"
    assert rep["summary"] == {"kept": 1, "broken": 1, "total": 2}
    broken = [o for o in rep["outcomes"] if not o["kept"]]
    assert broken[0]["command"] == "mytool check"


def test_all_kept_is_a_clean_status(tmp_path: Path) -> None:
    promises = [Promise(command="mytool --help", kind="console_script",
                        source="pyproject.toml")]
    outcomes = [{"command": "mytool --help", "exit_code": 0,
                 "timed_out": False, "output": "usage"}]
    rep = probe_report(tmp_path, promises, outcomes)
    assert rep["status"] == "promises_kept"


def test_a_server_promise_is_kept_by_staying_up(tmp_path: Path) -> None:
    """`gateway serve` never exits — and scoring it "broken" on timeout
    would mark every documented server as failing. For a long-running
    promise the contract is "starts and stays up", so surviving the grace
    period IS the promise being kept.

    Seen while extracting 16 real promises from a real project: three were
    servers. A gate with a systematic false positive gets ignored.
    """
    p = Promise(command="verimem gateway serve", kind="doc_command",
                source="README.md")
    assert p.expects_exit is False
    rep = probe_report(tmp_path, [p], [{
        "command": "verimem gateway serve", "exit_code": -1,
        "timed_out": True, "output": "listening on 8080",
    }])
    assert rep["status"] == "promises_kept"
    assert rep["outcomes"][0]["kept"] is True
    assert "stayed up" in rep["outcomes"][0].get("why", "").lower()


def test_a_server_that_dies_immediately_is_broken(tmp_path: Path) -> None:
    p = Promise(command="mytool serve", kind="doc_command",
                source="README.md")
    rep = probe_report(tmp_path, [p], [{
        "command": "mytool serve", "exit_code": 1, "timed_out": False,
        "output": "Traceback: port already in use",
    }])
    assert rep["status"] == "broken_promises"
    assert rep["outcomes"][0]["kept"] is False


@pytest.mark.parametrize("cmd,expects_exit", [
    ("mytool --help", True),
    ("mytool check", True),
    ("mytool gateway serve", False),
    ("mytool serve --port 8080", False),
    ("mytool start", False),
    ("mytool daemon", False),
    ("mytool watch src/", False),
    ("mytool console", False),
])
def test_long_running_promises_are_recognised(cmd: str,
                                              expects_exit: bool) -> None:
    assert Promise(command=cmd, kind="doc_command",
                   source="x").expects_exit is expects_exit


def test_a_timeout_is_a_broken_promise(tmp_path: Path) -> None:
    """For a promise that SHOULD terminate, a timeout is a failure. (The
    command here is deliberately not a server: `mytool serve` would now be
    judged by the stays-up rule instead, which is a different test.)"""
    promises = [Promise(command="mytool check --all", kind="doc_command",
                        source="README.md")]
    outcomes = [{"command": "mytool check --all", "exit_code": -1,
                 "timed_out": True, "output": ""}]
    rep = probe_report(tmp_path, promises, outcomes)
    assert rep["status"] == "broken_promises"
    assert rep["outcomes"][0]["kept"] is False
    assert "timed out" in rep["outcomes"][0]["why"].lower()


def test_unexecuted_promises_are_declared_not_assumed_kept(
    tmp_path: Path,
) -> None:
    """A promise with no outcome must never read as satisfied — that would
    be the same silent-pass this whole tool exists to prevent."""
    promises = [
        Promise(command="a --help", kind="console_script", source="p"),
        Promise(command="b --help", kind="console_script", source="p"),
    ]
    outcomes = [{"command": "a --help", "exit_code": 0, "timed_out": False,
                 "output": "ok"}]
    rep = probe_report(tmp_path, promises, outcomes)
    assert rep["summary"]["total"] == 2
    assert rep["summary"]["kept"] == 1
    assert rep["summary"].get("not_run") == 1
    assert rep["status"] == "incomplete"
