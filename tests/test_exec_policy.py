"""Tests for the execution POLICY — the layer that assumes the caller is hostile.

The threat model that shapes this file: this package is an MCP server, so
its tools are reachable by ANY MCP-capable agent — other instances of me,
Codex, Cursor, a cron job, an agent that itself carries a prompt
injection. `project_dir` and `test_path` arrive from that caller.

An earlier version of this design reasoned "the command comes from the
caller, not from the model, therefore injection has no leverage". That is
wrong: it moves the trust from the model to the caller without ever
establishing that the caller is trustworthy. With execution wired up, a
caller-supplied `project_dir` means running pytest on a directory of the
caller's choosing, and pytest runs that directory's own code.

So execution is gated by four independent things, each tested here:

  1. OFF BY DEFAULT. Mounting this server does not grant execution;
     an operator turns it on deliberately (CRITIC_ALLOW_EXEC=1).
  2. ROOT ALLOWLIST. Even when enabled, only paths under roots the
     OPERATOR named (CRITIC_EXEC_ROOTS) may be executed in — not
     wherever the caller points.
  3. The selector must be a plain pytest node id (see test_pinned_exec).
  4. Containment is by path components after resolution, never by string
     prefix, and symlinks cannot walk out.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from critic_orchestrator.exec_policy import (
    ExecPolicy,
    ExecPolicyError,
    policy_from_env,
)


def _repo(tmp_path: Path, name: str = "repo") -> Path:
    r = tmp_path / name
    (r / ".git").mkdir(parents=True)
    return r


# ---------------------------------------------------------------------------
# 1. Off by default
# ---------------------------------------------------------------------------

def test_execution_is_disabled_by_default(monkeypatch: pytest.MonkeyPatch,
                                           tmp_path: Path) -> None:
    monkeypatch.delenv("CRITIC_ALLOW_EXEC", raising=False)
    monkeypatch.delenv("CRITIC_EXEC_ROOTS", raising=False)
    pol = policy_from_env()
    assert pol.enabled is False
    with pytest.raises(ExecPolicyError) as exc:
        pol.check(_repo(tmp_path))
    assert "not enabled" in str(exc.value).lower()


@pytest.mark.parametrize("value", ["0", "false", "no", "", "off"])
def test_falsy_values_do_not_enable_execution(
    monkeypatch: pytest.MonkeyPatch, value: str,
) -> None:
    monkeypatch.setenv("CRITIC_ALLOW_EXEC", value)
    assert policy_from_env().enabled is False


def test_enabling_requires_explicit_roots(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Enabling without naming roots must not mean "anywhere". A
    fail-open default here would be the whole hole."""
    monkeypatch.setenv("CRITIC_ALLOW_EXEC", "1")
    monkeypatch.delenv("CRITIC_EXEC_ROOTS", raising=False)
    pol = policy_from_env()
    with pytest.raises(ExecPolicyError) as exc:
        pol.check(_repo(tmp_path))
    assert "root" in str(exc.value).lower()


# ---------------------------------------------------------------------------
# 2. Root allowlist
# ---------------------------------------------------------------------------

def test_a_repo_under_an_allowed_root_passes(tmp_path: Path) -> None:
    root = tmp_path / "work"
    repo = _repo(root)
    pol = ExecPolicy(enabled=True, roots=[root])
    assert pol.check(repo) == repo.resolve()


def test_a_repo_outside_every_root_is_refused(tmp_path: Path) -> None:
    allowed = tmp_path / "work"
    allowed.mkdir()
    outside = _repo(tmp_path / "elsewhere")
    pol = ExecPolicy(enabled=True, roots=[allowed])
    with pytest.raises(ExecPolicyError) as exc:
        pol.check(outside)
    assert "outside" in str(exc.value).lower()


def test_sibling_prefix_is_not_inside(tmp_path: Path) -> None:
    """`/work-evil` must not pass for root `/work`: a string prefix check
    would accept it. Same class as the sandbox path check."""
    allowed = tmp_path / "work"
    allowed.mkdir()
    sibling = _repo(tmp_path / "work-evil")
    pol = ExecPolicy(enabled=True, roots=[allowed])
    with pytest.raises(ExecPolicyError):
        pol.check(sibling)


def test_traversal_out_of_an_allowed_root_is_refused(tmp_path: Path) -> None:
    allowed = tmp_path / "work"
    (allowed / "inner").mkdir(parents=True)
    _repo(tmp_path / "secret")
    pol = ExecPolicy(enabled=True, roots=[allowed])
    with pytest.raises(ExecPolicyError):
        pol.check(allowed / "inner" / ".." / ".." / "secret")


def test_symlinked_repo_pointing_outside_is_refused(tmp_path: Path) -> None:
    allowed = tmp_path / "work"
    allowed.mkdir()
    outside = _repo(tmp_path / "outside")
    link = allowed / "link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not permitted in this environment")
    pol = ExecPolicy(enabled=True, roots=[allowed])
    with pytest.raises(ExecPolicyError):
        pol.check(link)


def test_a_non_repository_is_refused(tmp_path: Path) -> None:
    """Execution happens in a git worktree, so a non-repo cannot be
    isolated — and running pytest straight in a caller-named directory is
    precisely what this policy exists to prevent."""
    root = tmp_path / "work"
    plain = root / "plain"
    plain.mkdir(parents=True)
    pol = ExecPolicy(enabled=True, roots=[root])
    with pytest.raises(ExecPolicyError) as exc:
        pol.check(plain)
    assert "git" in str(exc.value).lower()


def test_multiple_roots_are_supported(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    repo_b = _repo(b)
    pol = ExecPolicy(enabled=True, roots=[a, b])
    assert pol.check(repo_b) == repo_b.resolve()


def test_roots_are_parsed_from_the_env_pathsep(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    import os
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    monkeypatch.setenv("CRITIC_ALLOW_EXEC", "1")
    monkeypatch.setenv("CRITIC_EXEC_ROOTS", os.pathsep.join([str(a), str(b)]))
    pol = policy_from_env()
    assert pol.enabled is True
    assert {p.resolve() for p in pol.roots} == {a.resolve(), b.resolve()}


def test_a_nonexistent_root_is_ignored_not_fatal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """An operator's stale entry must not disable a valid one, but must
    also not silently become "allow everything"."""
    import os
    good = tmp_path / "good"
    good.mkdir()
    repo = _repo(good)
    monkeypatch.setenv("CRITIC_ALLOW_EXEC", "1")
    monkeypatch.setenv(
        "CRITIC_EXEC_ROOTS",
        os.pathsep.join([str(tmp_path / "does_not_exist"), str(good)]),
    )
    pol = policy_from_env()
    assert pol.check(repo) == repo.resolve()


def test_policy_reason_is_actionable() -> None:
    """The refusal has to tell an operator what to set — a bare denial
    gets worked around instead of configured."""
    pol = ExecPolicy(enabled=False, roots=[])
    with pytest.raises(ExecPolicyError) as exc:
        pol.check(Path("."))
    msg = str(exc.value)
    assert "CRITIC_ALLOW_EXEC" in msg and "CRITIC_EXEC_ROOTS" in msg


# --------------------------------------------------------------------------
# A dropped root must not be dropped SILENTLY (design review, medium): a
# typo narrows the policy, and the operator then reads "outside every
# configured execution root" about a path they believe they configured.
# --------------------------------------------------------------------------

def test_a_nonexistent_root_is_remembered_not_just_dropped(
        tmp_path, monkeypatch) -> None:
    from critic_orchestrator.exec_policy import policy_from_env
    ghost = str(tmp_path / "typo_dir_that_does_not_exist")
    monkeypatch.setenv("CRITIC_ALLOW_EXEC", "1")
    monkeypatch.setenv("CRITIC_EXEC_ROOTS", ghost)
    policy = policy_from_env()
    assert policy.roots == []
    assert ghost in policy.invalid_roots


def test_the_refusal_names_the_dropped_entry(tmp_path, monkeypatch) -> None:
    from critic_orchestrator.exec_policy import ExecPolicyError, policy_from_env
    ghost = str(tmp_path / "nope")
    monkeypatch.setenv("CRITIC_ALLOW_EXEC", "1")
    monkeypatch.setenv("CRITIC_EXEC_ROOTS", ghost)
    with pytest.raises(ExecPolicyError) as exc:
        policy_from_env().check(tmp_path)
    assert "nope" in str(exc.value)
    assert "not existing directories" in str(exc.value) \
        or "not an existing director" in str(exc.value) \
        or "dropped" in str(exc.value)


def test_a_valid_root_still_works_alongside_a_typo(
        tmp_path, monkeypatch) -> None:
    """A stale entry must never disable the valid ones."""
    import os as _os
    from critic_orchestrator.exec_policy import policy_from_env
    good = tmp_path / "repo"
    good.mkdir()
    (good / ".git").mkdir()
    ghost = str(tmp_path / "typo")
    monkeypatch.setenv("CRITIC_ALLOW_EXEC", "1")
    monkeypatch.setenv("CRITIC_EXEC_ROOTS",
                       _os.pathsep.join([ghost, str(tmp_path)]))
    policy = policy_from_env()
    assert policy.check(good) == good.resolve()
    assert ghost in policy.invalid_roots
